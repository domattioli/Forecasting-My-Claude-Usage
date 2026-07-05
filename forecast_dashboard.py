#!/usr/bin/env python3
"""
High-DPI 6-panel Bayesian forecast dashboard for Claude usage.
Reuses pipeline from forecast_usage.py (load_data, fit_conjugate_hurdle, get_price, etc.)
"""
import sys
import os
import sqlite3
import json
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add forecast_usage to path
sys.path.insert(0, '/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from forecast_usage import load_data, load_desktop_data, fit_conjugate_hurdle, PRICING, get_price


def monte_carlo_forecast(posterior, max_date, n_draws=6000):
    """
    Generate full (n_draws x 30) future-cost matrix for shared draws across all panels.
    posterior: dict from fit_conjugate_hurdle()
    max_date: last date in historical data
    n_draws: number of MC draws

    Returns: (M, forecast_dates) where M is (n_draws x 30) matrix of daily costs
    """
    np.random.seed(42)

    alpha_wd, beta_wd = posterior['p_weekday_post']
    alpha_we, beta_we = posterior['p_weekend_post']

    mu_n = posterior['mu_n']
    kappa_n = posterior['kappa_n']
    alpha_n = posterior['alpha_n']
    beta_n = posterior['beta_n']
    delta = posterior['weekend_delta_shrunk']

    horizon = 30
    forecast_dates = pd.date_range(start=max_date + timedelta(days=1), periods=horizon, freq='D')

    # Initialize matrix (n_draws x 30)
    M = np.zeros((n_draws, horizon))
    M_exp = np.zeros((n_draws, horizon))  # Expected values for each day

    for draw_idx in range(n_draws):
        # Sample p_weekday, p_weekend from Beta posteriors
        p_wd = np.random.beta(alpha_wd, beta_wd)
        p_we = np.random.beta(alpha_we, beta_we)

        # Sample (sigma^2, mu) from NIG posterior
        sigma2 = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
        sigma = np.sqrt(sigma2)
        mu = np.random.normal(mu_n, np.sqrt(sigma2 / kappa_n))

        for day_idx, d in enumerate(forecast_dates):
            w = d.weekday()

            # Pick group and occurrence probability
            if w < 5:  # Weekday
                p_group = p_wd
                group_shift = -0.5 * delta
            else:  # Weekend
                p_group = p_we
                group_shift = 0.5 * delta

            # Analytic expected value
            exp_val = p_group * np.exp(mu + group_shift + 0.5 * sigma2)
            M_exp[draw_idx, day_idx] = exp_val

            # Sample occurrence z ~ Bernoulli(p_group)
            z = np.random.binomial(1, p_group)

            if z == 1:
                # Sample log_y ~ Normal(mu + group_shift, sigma^2)
                log_y = np.random.normal(mu + group_shift, sigma)
                y = np.exp(log_y)
            else:
                y = 0.0

            M[draw_idx, day_idx] = y

    return M, M_exp, forecast_dates


def get_model_family(model_key):
    """Extract model family from pricing key."""
    if not model_key:
        return "Unknown"
    if "opus" in model_key.lower():
        return "Opus"
    elif "sonnet" in model_key.lower():
        return "Sonnet"
    elif "haiku" in model_key.lower():
        return "Haiku"
    elif "fable" in model_key.lower():
        return "Fable"
    else:
        return "Other"


def get_model_spend():
    """Query usage.db and aggregate total USD by model family."""
    db_path = os.path.expanduser("~/.claude/usage.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model, input_tokens, output_tokens,
               cache_creation_tokens, cache_read_tokens
        FROM turns
    """)
    rows = cursor.fetchall()
    conn.close()

    family_costs = {}

    for model, inp_tok, out_tok, cache_cre_tok, cache_read_tok in rows:
        if model is None:
            continue
        price = get_price(model)
        if price is None:
            continue

        # Compute cost (USD)
        cost = (inp_tok * price['input'] +
                out_tok * price['output'] +
                cache_cre_tok * price['cache_write'] +
                cache_read_tok * price['cache_read']) / 1e6

        family = get_model_family(model)
        if family not in family_costs:
            family_costs[family] = 0.0
        family_costs[family] += cost

    return family_costs


def plot_dashboard(df, desktop, posterior, M, M_exp, forecast_dates, max_cli_date):
    """Create 6-panel dashboard."""
    plt.style.use('seaborn-v0_8-darkgrid')

    # Colors
    cli_color = '#4C72B0'
    desktop_color = '#55A868'
    forecast_color = '#DD8452'
    expected_color = '#C44E52'

    # Color palette for per-model spend bars
    model_colors = {
        'Opus': '#1f77b4',
        'Sonnet': '#ff7f0e',
        'Haiku': '#2ca02c',
        'Fable': '#d62728',
        'Other': '#9467bd'
    }

    # Build combined history (CLI + desktop, aligned to same date range)
    min_hist_date = df['date'].min()
    if desktop:
        min_hist_date = min(min_hist_date, min(desktop.keys()))

    full_date_range = pd.date_range(start=min_hist_date, end=max_cli_date, freq='D')
    combined_data = []
    for d in full_date_range:
        date_obj = d.date()
        cli_cost = float(df[df['date'] == date_obj]['cost'].values[0]) if date_obj in df['date'].values else 0.0
        desktop_cost = desktop.get(date_obj, 0.0)
        combined_data.append({
            'date': d,
            'date_obj': date_obj,
            'cli_cost': cli_cost,
            'desktop_cost': desktop_cost,
            'weekday': d.weekday()
        })
    df_combined = pd.DataFrame(combined_data)

    # Figure: (18, 10), 2 rows x 3 cols, with top padding for suptitle
    fig = plt.figure(figsize=(18, 10), constrained_layout=False)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, top=0.94, bottom=0.08, left=0.06, right=0.96)

    max_cli_date_str = max_cli_date.strftime('%Y-%m-%d')
    fig.suptitle(f'Claude usage -- Bayesian forecast dashboard (as of {max_cli_date_str})',
                 fontsize=15, fontweight='bold', y=0.98)

    # ========== PANEL 1: History + Forecast Fan ==========
    ax1 = fig.add_subplot(gs[0, 0])

    hist_dates = df_combined['date'].values
    cli_costs = df_combined['cli_cost'].values
    desktop_costs = df_combined['desktop_cost'].values

    ax1.bar(hist_dates, cli_costs, color=cli_color, alpha=0.85, width=0.8, label='CLI')
    ax1.bar(hist_dates, desktop_costs, bottom=cli_costs, color=desktop_color, alpha=0.85, width=0.8, label='Desktop')

    # Forecast boundary
    boundary = pd.Timestamp(max_cli_date)
    ax1.axvline(boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    # Forecast quantiles from M
    q5 = np.percentile(M, 5, axis=0)
    q25 = np.percentile(M, 25, axis=0)
    q50 = np.percentile(M, 50, axis=0)
    q75 = np.percentile(M, 75, axis=0)
    q95 = np.percentile(M, 95, axis=0)
    exp_mean = np.mean(M_exp, axis=0)

    forecast_dates_plot = forecast_dates.to_pydatetime()
    ax1.fill_between(forecast_dates_plot, q5, q95, color=forecast_color, alpha=0.18, label='90% interval')
    ax1.fill_between(forecast_dates_plot, q25, q75, color=forecast_color, alpha=0.33, label='50% interval')
    ax1.plot(forecast_dates_plot, q50, color=forecast_color, marker='o', markersize=4, linewidth=2, label='Median')
    ax1.plot(forecast_dates_plot, exp_mean, color=expected_color, linestyle='--', linewidth=2, label='Expected value')

    ax1.set_title('Daily cost + 30-day forecast', fontsize=12, fontweight='bold')
    ax1.set_ylabel('USD/day', fontsize=11)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.tick_params(axis='x', rotation=45)
    fig.autofmt_xdate()

    # Annotation box - moved to bottom-right to avoid legend collision
    total_30d = M.sum(axis=1)
    total_30d_mean = total_30d.mean()
    total_30d_q5 = np.percentile(total_30d, 5)
    total_30d_q95 = np.percentile(total_30d, 95)
    textstr = f"30d total: ${total_30d_mean:.0f}\n(90% CI ${total_30d_q5:.0f}-${total_30d_q95:.0f})"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax1.text(0.98, 0.35, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    # ========== PANEL 2: Cumulative Spend ==========
    ax2 = fig.add_subplot(gs[0, 1])

    M_cumsum = np.cumsum(M, axis=1)  # (n_draws x 30)
    cum_q5 = np.percentile(M_cumsum, 5, axis=0)
    cum_q50 = np.percentile(M_cumsum, 50, axis=0)
    cum_q95 = np.percentile(M_cumsum, 95, axis=0)

    days_ahead = np.arange(1, 31)
    ax2.fill_between(days_ahead, cum_q5, cum_q95, color=forecast_color, alpha=0.25)
    ax2.plot(days_ahead, cum_q50, color=forecast_color, marker='o', markersize=3, linewidth=2, label='Median')

    ax2.set_title('Cumulative 30-day spend', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Day ahead', fontsize=11)
    ax2.set_ylabel('Cumulative USD', fontsize=11)
    ax2.legend(fontsize=9)

    # Annotate day 30
    ax2.text(30, cum_q50[-1], f' ${cum_q50[-1]:.0f}', fontsize=9, va='center')
    ax2.text(30, cum_q5[-1], f' (${cum_q5[-1]:.0f}-${cum_q95[-1]:.0f})', fontsize=8, va='center', style='italic')

    # ========== PANEL 3: 30-Day Total Posterior ==========
    ax3 = fig.add_subplot(gs[0, 2])

    # Clip x-range at p99 to avoid extreme tail blowout
    p99_val = np.percentile(total_30d, 99)
    max_val = total_30d.max()

    # Create histogram only within [0, p99]
    ax3.hist(total_30d, bins=60, density=True, color=forecast_color, alpha=0.7, edgecolor='black', linewidth=0.5, range=(0, p99_val))
    ax3.set_xlim(0, p99_val)

    # Lines (only draw if within xlim)
    ax3.axvline(total_30d_mean, color=expected_color, linestyle='-', linewidth=2, label=f'Mean: ${total_30d_mean:.0f}')
    med_30d = np.median(total_30d)
    ax3.axvline(med_30d, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: ${med_30d:.0f}')
    if total_30d_q5 <= p99_val:
        ax3.axvline(total_30d_q5, color='darkred', linestyle=':', linewidth=1.5, alpha=0.6)
    if total_30d_q95 <= p99_val:
        ax3.axvline(total_30d_q95, color='darkred', linestyle=':', linewidth=1.5, alpha=0.6, label='90% CI')

    # Shade 90% CI region (clipped)
    q5_clipped = min(total_30d_q5, p99_val)
    q95_clipped = min(total_30d_q95, p99_val)
    ax3.axvspan(q5_clipped, q95_clipped, alpha=0.1, color='darkred')

    ax3.set_title('30-day total - posterior', fontsize=12, fontweight='bold')
    ax3.set_xlabel('USD', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.legend(fontsize=9, loc='upper left')

    # Annotation about tail clipping (moved below legend to avoid overlap)
    tail_note = f"x-axis clipped at p99 = ${p99_val:.0f}\nrare tail reaches ${max_val/1000:.0f}k"
    ax3.text(0.97, 0.55, tail_note, transform=ax3.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), style='italic')

    # ========== PANEL 4: Occurrence Posteriors (Weekday vs Weekend) ==========
    ax4 = fig.add_subplot(gs[1, 0])

    alpha_wd, beta_wd = posterior['p_weekday_post']
    alpha_we, beta_we = posterior['p_weekend_post']

    # Sample Beta draws and KDE-lite via histogram
    x_occ = np.linspace(0, 1, 200)
    samples_wd = np.random.beta(alpha_wd, beta_wd, 100000)
    samples_we = np.random.beta(alpha_we, beta_we, 100000)

    # Plot histograms as density
    ax4.hist(samples_wd, bins=50, density=True, alpha=0.6, color=cli_color, label=f'Weekday (mu={samples_wd.mean():.3f})', edgecolor='black', linewidth=0.3)
    ax4.hist(samples_we, bins=50, density=True, alpha=0.6, color=desktop_color, label=f'Weekend (mu={samples_we.mean():.3f})', edgecolor='black', linewidth=0.3)

    ax4.set_title('P(active day) - weekday vs weekend', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Probability', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.legend(fontsize=9, loc='upper right')

    # Annotation at BOTTOM to avoid legend collision
    if posterior['n_weekday'] < 30:
        note = f"Heavy overlap => weekday effect not resolved (n={posterior['n_weekday']})"
        props_note = dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
        ax4.text(0.5, 0.05, note, transform=ax4.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='center', bbox=props_note, style='italic')

    # ========== PANEL 5: Log Daily Cost (Active) + Fitted Lognormal ==========
    ax5 = fig.add_subplot(gs[1, 1])

    active_costs_log = posterior['active_costs_log']
    n_active = len(active_costs_log)

    # Histogram of log costs
    ax5.hist(active_costs_log, bins=15, density=True, color=forecast_color, alpha=0.7, edgecolor='black', linewidth=0.5, label='Active days')

    # Fitted Normal curve on log scale
    mu_fitted = posterior['mu_n']
    sigma_fitted = np.sqrt(posterior['beta_n'] / (posterior['alpha_n'] - 1)) if posterior['alpha_n'] > 1 else 0.5

    x_log = np.linspace(active_costs_log.min() - 0.5, active_costs_log.max() + 0.5, 200)
    pdf_vals = stats.norm.pdf(x_log, mu_fitted, sigma_fitted)
    ax5.plot(x_log, pdf_vals, color=expected_color, linewidth=2.5, label='Fitted Normal')

    ax5.set_title('Log daily cost | active + lognormal', fontsize=12, fontweight='bold')
    ax5.set_xlabel('log(USD)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.legend(fontsize=9)

    # Annotation
    median_active_cost = np.exp(mu_fitted)
    note_active = f"n={n_active}, implied median=${median_active_cost:.2f}"
    props_active = dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
    ax5.text(0.98, 0.97, note_active, transform=ax5.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props_active, family='monospace')

    # ========== PANEL 6: Per-Model Spend (All-time) ==========
    ax6 = fig.add_subplot(gs[1, 2])

    family_costs = get_model_spend()

    if family_costs:
        # Sort descending by cost
        sorted_families = sorted(family_costs.items(), key=lambda x: x[1], reverse=True)
        families = [f[0] for f in sorted_families]
        costs = [f[1] for f in sorted_families]
        total_cost = sum(costs)

        # Colors for each bar
        bar_colors = [model_colors.get(f, '#9467bd') for f in families]

        # Create horizontal bar chart
        y_pos = np.arange(len(families))
        bars = ax6.barh(y_pos, costs, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Label bars with "$X (Y%)" format
        for i, (bar, cost) in enumerate(zip(bars, costs)):
            pct = 100 * cost / total_cost if total_cost > 0 else 0
            label = f"${cost:.0f} ({pct:.1f}%)"
            ax6.text(cost + 0.01 * total_cost, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=9, family='monospace')

        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(families, fontsize=10)
        ax6.set_xlabel('USD', fontsize=11)
        ax6.set_title('CLI spend by model family (all-time)', fontsize=12, fontweight='bold')
        ax6.invert_yaxis()

        # Print to console
        print("\nPer-Model Spend Breakdown:")
        print("-" * 50)
        for family, cost in sorted_families:
            pct = 100 * cost / total_cost if total_cost > 0 else 0
            print(f"  {family:12s} ${cost:10.2f} ({pct:5.1f}%)")
        print(f"  {'TOTAL':12s} ${total_cost:10.2f} (100.0%)")
        print("-" * 50)
    else:
        ax6.text(0.5, 0.5, 'No spend data available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=11)
        ax6.set_title('CLI spend by model family (all-time)', fontsize=12, fontweight='bold')

    # ========== SUMMARY: Print to console ==========
    print("\n" + "="*70)
    print("FORECAST DASHBOARD SUMMARY")
    print("="*70)
    print(f"Fit window (CLI only): {len(df)} days ({df['date'].min()} to {df['date'].max()})")
    print(f"  Active days: {np.sum(df['cost'].values > 0)} ({100*np.sum(df['cost'].values > 0)/len(df):.1f}%)")
    print()
    print(f"Posterior estimates:")
    print(f"  Weekday occurrence: Beta{posterior['p_weekday_post']}")
    print(f"  Weekend occurrence: Beta{posterior['p_weekend_post']}")
    print(f"  Log-amount: mu={mu_fitted:.3f}, sigma~{sigma_fitted:.3f}")
    print(f"  Weekend shift (shrunk): {posterior['weekend_delta_shrunk']:.3f}")
    print()
    print(f"30-day forecast (n_draws={M.shape[0]}):")
    print(f"  Total: ${total_30d_mean:.0f} (90% CI ${total_30d_q5:.0f}-${total_30d_q95:.0f})")
    print(f"  Mean per day: ${total_30d_mean/30:.2f}")
    print()
    print(f"Panel 3 (30-day posterior) clipping:")
    print(f"  p99 threshold: ${p99_val:.0f}")
    print(f"  Max draw value: ${max_val:.0f}")
    print()
    print("="*70)

    plt.savefig('/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_dashboard.png',
                dpi=200, bbox_inches='tight')
    print(f"Dashboard saved to usage_dashboard.png (dpi=200)")
    print("="*70)


def main():
    print("\nLoading data...")
    np.random.seed(42)

    df = load_data()
    desktop = load_desktop_data()

    max_cli_date = df['date'].max()
    print(f"CLI data: {len(df)} days, max date {max_cli_date}")
    print(f"Desktop data: {len(desktop)} dates")

    # Fit
    print("\nFitting conjugate Bayesian hurdle model...")
    y = df['cost'].values
    w = df['weekday'].values
    posterior = fit_conjugate_hurdle(y, w)
    print(f"  Posterior params computed")

    # Monte Carlo
    print("\nRunning Monte Carlo forecast (n_draws=6000)...")
    M, M_exp, forecast_dates = monte_carlo_forecast(posterior, max_cli_date, n_draws=6000)
    print(f"  Generated forecast matrix shape: {M.shape}")

    # Plot
    print("\nCreating dashboard...")
    plot_dashboard(df, desktop, posterior, M, M_exp, forecast_dates, max_cli_date)

    # Verify file
    import os
    fpath = '/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_dashboard.png'
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat()
        print(f"\nFile verified: {size_mb:.2f} MB, mtime={mtime}")

    print("\nDone!")


if __name__ == '__main__':
    main()
