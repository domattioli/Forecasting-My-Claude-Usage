#!/usr/bin/env python3
import sqlite3
import json
import math
import os
import glob
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configuration - reuse from forecast_usage.py
PRICING = {
    'claude-fable-5':    dict(input=10.00, output=50.00, cache_write=12.50, cache_read=1.00),
    'claude-mythos-5':   dict(input=10.00, output=50.00, cache_write=12.50, cache_read=1.00),
    'claude-opus-4-8':   dict(input=5.00,  output=25.00, cache_write=6.25,  cache_read=0.50),
    'claude-opus-4-7':   dict(input=5.00,  output=25.00, cache_write=6.25,  cache_read=0.50),
    'claude-opus-4-6':   dict(input=5.00,  output=25.00, cache_write=6.25,  cache_read=0.50),
    'claude-sonnet-4-7': dict(input=3.00,  output=15.00, cache_write=3.75,  cache_read=0.30),
    'claude-sonnet-4-6': dict(input=3.00,  output=15.00, cache_write=3.75,  cache_read=0.30),
    'claude-sonnet-4-5': dict(input=3.00,  output=15.00, cache_write=3.75,  cache_read=0.30),
    'claude-haiku-4-7':  dict(input=1.00,  output=5.00,  cache_write=1.25,  cache_read=0.10),
    'claude-haiku-4-6':  dict(input=1.00,  output=5.00,  cache_write=1.25,  cache_read=0.10),
    'claude-haiku-4-5':  dict(input=1.00,  output=5.00,  cache_write=1.25,  cache_read=0.10),
}

def get_price(model):
    """Match model by longest key prefix."""
    if model is None:
        return None
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        if model.startswith(key):
            return PRICING[key]
    return None

def model_to_family(model_str):
    """Map model string to family: Opus/Sonnet/Haiku/Fable."""
    if model_str is None:
        return None
    if 'opus' in model_str.lower():
        return 'Opus'
    elif 'sonnet' in model_str.lower():
        return 'Sonnet'
    elif 'haiku' in model_str.lower():
        return 'Haiku'
    elif 'fable' in model_str.lower() or 'mythos' in model_str.lower():
        return 'Fable'
    return None

def load_data():
    """Load usage data from SQLite, compute daily costs per model family, return dict of DataFrames."""
    db_path = os.path.expanduser("~/.claude/usage.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, model, input_tokens, output_tokens,
               cache_creation_tokens, cache_read_tokens
        FROM turns
    """)
    rows = cursor.fetchall()
    conn.close()

    # daily_costs_per_family: dict of dict
    # daily_costs_per_family[family][date] = cost_usd
    daily_costs_per_family = {family: {} for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']}
    chicago_tz = ZoneInfo("America/Chicago")

    for timestamp_str, model, inp_tok, out_tok, cache_cre_tok, cache_read_tok in rows:
        if model is None:
            continue
        price = get_price(model)
        if price is None:
            continue

        family = model_to_family(model)
        if family is None:
            continue

        # Parse timestamp
        iso_str = timestamp_str.replace('Z', '+00:00')
        dt_utc = datetime.fromisoformat(iso_str)
        dt_chicago = dt_utc.astimezone(chicago_tz)
        date = dt_chicago.date()

        # Compute cost
        cost = (inp_tok * price['input'] +
                out_tok * price['output'] +
                cache_cre_tok * price['cache_write'] +
                cache_read_tok * price['cache_read']) / 1e6

        if date not in daily_costs_per_family[family]:
            daily_costs_per_family[family][date] = 0.0
        daily_costs_per_family[family][date] += cost

    # Build DataFrames per family over fit window
    # Fit window: 2026-06-02 to max date in data
    all_dates = set()
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        all_dates.update(daily_costs_per_family[family].keys())

    if not all_dates:
        raise ValueError("No data loaded from database")

    min_date = datetime(2026, 6, 2).date()
    max_date = max(all_dates)

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    dfs = {}
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        data = []
        for d in date_range:
            date_obj = d.date()
            cost = daily_costs_per_family[family].get(date_obj, 0.0)
            data.append({'date': date_obj, 'cost': cost})
        dfs[family] = pd.DataFrame(data)

    return dfs, min_date, max_date

def fit_conjugate_family(y):
    """
    Fit closed-form conjugate Bayesian hurdle model for a model family (NO weekday split).
    y: daily costs (N,)

    Returns: dict with posterior params and metadata
      - active_days: number of positive-cost days
      - total_hist: total historical cost
      - p_post: (alpha_post, beta_post) Beta posterior for occurrence
      - mu_n, kappa_n, alpha_n, beta_n: NIG posterior on log-costs
      - low_confidence: True if n_active < 3
    """
    n_days = len(y)
    n_active = np.sum(y > 0)

    if n_active == 0:
        raise ValueError("No active days in family")

    # Occurrence: Beta-Binomial conjugate (Beta prior (1,1) = Jeffreys)
    p_post = (1 + n_active, 1 + n_days - n_active)

    # Amount: Normal-Inverse-Gamma on log-costs (active days only)
    active_costs = y[y > 0]
    log_costs = np.log(active_costs)

    total_hist = float(np.sum(active_costs))

    # NIG prior
    mu0 = np.mean(log_costs)
    kappa0 = 1.0
    alpha0 = 2.0
    var_log = np.var(log_costs, ddof=1) if n_active > 1 else 2.0
    beta0 = 0.5 * var_log * alpha0

    # Posterior
    n = len(log_costs)
    mean_log = np.mean(log_costs)

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean_log) / kappa_n
    alpha_n = alpha0 + n / 2.0

    ssd = np.sum((log_costs - mean_log) ** 2)
    beta_n = beta0 + 0.5 * ssd + 0.5 * kappa0 * n / kappa_n * (mean_log - mu0) ** 2

    low_confidence = (n_active < 3)

    return {
        'active_days': n_active,
        'total_hist': total_hist,
        'p_post': p_post,
        'mu_n': mu_n,
        'kappa_n': kappa_n,
        'alpha_n': alpha_n,
        'beta_n': beta_n,
        'low_confidence': low_confidence,
        'n_active': n_active,
    }

def forecast_30days_per_family_mc(posterior, max_date, n_draws=6000):
    """
    Monte Carlo forecast 30 days ahead for one model family.
    Returns: M_fam (n_draws x 30) matrix of costs
    """
    alpha_p, beta_p = posterior['p_post']
    mu_n = posterior['mu_n']
    kappa_n = posterior['kappa_n']
    alpha_n = posterior['alpha_n']
    beta_n = posterior['beta_n']

    horizon = 30
    forecast_dates = pd.date_range(start=max_date + timedelta(days=1), periods=horizon, freq='D')

    # M_fam: n_draws x horizon
    M_fam = np.zeros((n_draws, horizon))

    for draw_idx in range(n_draws):
        # Sample occurrence probability
        p = np.random.beta(alpha_p, beta_p)

        # Sample (sigma^2, mu) from NIG posterior
        sigma2 = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
        mu = np.random.normal(mu_n, np.sqrt(sigma2 / kappa_n))

        for day_idx in range(horizon):
            # Bernoulli: occurs with probability p
            z = np.random.binomial(1, p)

            if z == 1:
                # Sample cost from lognormal
                log_y = np.random.normal(mu, np.sqrt(sigma2))
                y = np.exp(log_y)
            else:
                y = 0.0

            M_fam[draw_idx, day_idx] = y

    return M_fam

def plot_permodel_forecast(dfs, posteriors, M_dict, max_date, min_date):
    """
    Create comprehensive per-model forecast visualization.
    dfs: dict of family -> DataFrame with historical data
    posteriors: dict of family -> posterior params
    M_dict: dict of family -> (n_draws x 30) forecast matrix
    max_date: last date in history
    min_date: first date in fit window
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Color map for families
    colors = {'Opus': '#4C72B0', 'Sonnet': '#55A868', 'Haiku': '#DD8452', 'Fable': '#C44E52'}
    families = ['Opus', 'Sonnet', 'Haiku', 'Fable']

    # =========== PANEL A (top-left, span 2 cols): STACKED history + stacked forecast ===========
    ax_a = fig.add_subplot(gs[0, :2])

    # Historical dates
    hist_dates = dfs['Opus']['date'].values
    date_range_hist = [pd.Timestamp(d) for d in hist_dates]

    # Build stacked historical data
    hist_data_by_family = {}
    for family in families:
        hist_data_by_family[family] = dfs[family]['cost'].values

    # Plot stacked historical bars
    bottom = np.zeros(len(hist_dates))
    for family in families:
        ax_a.bar(date_range_hist, hist_data_by_family[family], bottom=bottom,
                 label=family, color=colors[family], alpha=0.85, width=0.8)
        bottom += hist_data_by_family[family]

    # Forecast dates
    forecast_dates = pd.date_range(start=max_date + timedelta(days=1), periods=30, freq='D')
    forecast_dates_ts = [pd.Timestamp(d) for d in forecast_dates]

    # Aggregate forecast across all families (from M_dict)
    # For each day, sum across families to get aggregate daily distribution
    # M_dict[family] is (n_draws x 30)
    n_draws = list(M_dict.values())[0].shape[0]
    M_agg = np.zeros((n_draws, 30))
    for family in families:
        M_agg += M_dict[family]

    # Compute per-day aggregates: median and 90% band
    agg_median = np.percentile(M_agg, 50, axis=0)
    agg_q5 = np.percentile(M_agg, 5, axis=0)
    agg_q95 = np.percentile(M_agg, 95, axis=0)

    # Plot aggregate 90% band (light fill, BEHIND the stacked areas)
    ax_a.fill_between(forecast_dates_ts, agg_q5, agg_q95, color='gray', alpha=0.15, zorder=1)

    # Stacked expected-value forecast: each family's per-day median expected cost
    # Compute per-family expected value: E[cost|family] = p_post_mean * exp(mu_n + 0.5*sigma2)
    bottom_forecast = np.zeros(30)
    forecast_handles = []
    for family in families:
        M_fam = M_dict[family]
        fam_median = np.median(M_fam, axis=0)  # Median of each day's distribution
        h = ax_a.fill_between(forecast_dates_ts, bottom_forecast, bottom_forecast + fam_median,
                          color=colors[family], alpha=0.75, label=f'{family} (forecast)', zorder=2)
        forecast_handles.append(h)
        bottom_forecast += fam_median

    # Aggregate median line
    ax_a.plot(forecast_dates_ts, agg_median, color='black', linewidth=2.5, label='Aggregate median',
              marker='o', markersize=2, zorder=3)

    # Boundary line
    boundary = pd.Timestamp(max_date)
    ax_a.axvline(boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax_a.set_title('Per-model history + stacked forecast (aggregate band)', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('USD/day')
    ax_a.set_xlabel('Date')
    # Build custom legend: history bars + forecast areas + band + aggregate line
    hist_handles = [plt.Rectangle((0,0),1,1, facecolor=colors[f], alpha=0.85, label=f'{f} (history)')
                    for f in families]
    band_handle = plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.15, label='Aggregate 90% band')
    agg_line_handle = plt.Line2D([0], [0], color='black', linewidth=2.5, marker='o', markersize=2, label='Aggregate median')
    all_handles = hist_handles + forecast_handles + [band_handle, agg_line_handle]
    all_labels = [f'{f} (history)' for f in families] + [f'{f} (forecast)' for f in families] + \
                 ['Aggregate 90% band', 'Aggregate median']
    ax_a.legend(handles=all_handles, labels=all_labels, loc='upper left', fontsize=8, ncol=2)
    ax_a.tick_params(axis='x', rotation=45)
    fig.autofmt_xdate()

    # =========== PANEL B (top-right): 30-day TOTAL per family ===========
    ax_b = fig.add_subplot(gs[0, 2])

    totals_30d = {}
    q5_dict = {}
    q95_dict = {}
    for family in families:
        M_fam = M_dict[family]
        totals_per_draw = M_fam.sum(axis=1)  # Sum across 30 days per draw
        totals_30d[family] = np.median(totals_per_draw)
        q5_dict[family] = np.percentile(totals_per_draw, 5)
        q95_dict[family] = np.percentile(totals_per_draw, 95)

    y_pos = np.arange(len(families))
    medians = [totals_30d[f] for f in families]
    errors = np.array([
        [totals_30d[f] - q5_dict[f] for f in families],
        [q95_dict[f] - totals_30d[f] for f in families]
    ])

    bars = ax_b.barh(y_pos, medians, xerr=errors, color=[colors[f] for f in families], alpha=0.85,
                     error_kw={'elinewidth': 1.5, 'capsize': 3})

    # Hatch Fable if low confidence
    if posteriors['Fable']['low_confidence']:
        bars[-1].set_hatch('///')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(families)
    ax_b.set_xlabel('30-day total (USD)')
    ax_b.set_title('30-day forecast by model\n(median, 90% CI)', fontsize=11, fontweight='bold')

    # Place value labels and confidence notes to the RIGHT, outside bars
    for i, f in enumerate(families):
        med_val = medians[i]
        ci_hi = q95_dict[f]
        # Dollar label to the right of error bar
        ax_b.text(ci_hi + 8, i, f'${med_val:.0f}', va='center', fontsize=9, fontweight='bold')
        # Confidence note for Fable
        if posteriors[f]['low_confidence']:
            ax_b.text(ci_hi + 8, i - 0.25, '(n=2, low-conf)', va='top', fontsize=8, style='italic', color='gray')

    # =========== PANEL C (bottom-left): Aggregate 30-day posterior ===========
    ax_c = fig.add_subplot(gs[1, 0])

    agg_total_30d = M_agg.sum(axis=1)  # Sum across 30 days per draw
    p99 = np.percentile(agg_total_30d, 99)
    agg_median_30d = np.median(agg_total_30d)
    agg_mean_30d = np.mean(agg_total_30d)
    agg_q5_30d = np.percentile(agg_total_30d, 5)
    agg_q95_30d = np.percentile(agg_total_30d, 95)

    ax_c.hist(agg_total_30d[agg_total_30d <= p99], bins=40, color='#4C72B0', alpha=0.7, edgecolor='black')
    ax_c.axvline(agg_mean_30d, color='red', linestyle='--', linewidth=2, label=f'Mean: ${agg_mean_30d:.0f}')
    ax_c.axvline(agg_median_30d, color='green', linestyle='-', linewidth=2, label=f'Median: ${agg_median_30d:.0f}')
    ax_c.axvline(agg_q5_30d, color='orange', linestyle=':', linewidth=1.5, label=f'90% CI: ${agg_q5_30d:.0f}-${agg_q95_30d:.0f}')
    ax_c.axvline(agg_q95_30d, color='orange', linestyle=':', linewidth=1.5)

    ax_c.set_xlabel('30-day total (USD)')
    ax_c.set_ylabel('Frequency')
    ax_c.set_title('Aggregate 30-day total\n(cross-check vs pooled)', fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=9)

    # Annotation: pooled reference
    textstr = f'Pooled ref: $254'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_c.text(0.98, 0.97, textstr, transform=ax_c.transAxes, fontsize=9,
              verticalalignment='top', horizontalalignment='right', bbox=props)

    # =========== PANEL D (bottom-middle): Per-family occurrence rate ===========
    ax_d = fig.add_subplot(gs[1, 1])

    p_means = []
    p_q5s = []
    p_q95s = []
    for family in families:
        alpha_p, beta_p = posteriors[family]['p_post']
        # Posterior mean of Beta(alpha, beta) is alpha / (alpha + beta)
        p_mean = alpha_p / (alpha_p + beta_p)
        # For 90% CI, sample from Beta
        p_samples = np.random.beta(alpha_p, beta_p, 10000)
        p_q5 = np.percentile(p_samples, 5)
        p_q95 = np.percentile(p_samples, 95)
        p_means.append(p_mean)
        p_q5s.append(p_q5)
        p_q95s.append(p_q95)

    y_pos_p = np.arange(len(families))
    errors_p = np.array([
        [p_means[i] - p_q5s[i] for i in range(len(families))],
        [p_q95s[i] - p_means[i] for i in range(len(families))]
    ])

    ax_d.barh(y_pos_p, p_means, xerr=errors_p, color=[colors[f] for f in families], alpha=0.85,
              error_kw={'elinewidth': 1.5, 'capsize': 3})
    ax_d.set_yticks(y_pos_p)
    ax_d.set_yticklabels(families)
    ax_d.set_xlabel('Activity rate (Pr[active])')
    ax_d.set_title('Per-model activity rate\n(posterior mean, 90% CI)', fontsize=11, fontweight='bold')
    ax_d.set_xlim(0, 1)

    # =========== PANEL E (bottom-right): Active days summary ===========
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')

    summary_text = "Per-family summary (fit window):\n\n"
    for family in families:
        n_act = posteriors[family]['n_active']
        total = posteriors[family]['total_hist']
        summary_text += f"{family}:\n"
        summary_text += f"  Active days: {n_act}\n"
        summary_text += f"  Total cost: ${total:.0f}\n\n"

    ax_e.text(0.05, 0.95, summary_text, transform=ax_e.transAxes, fontsize=9,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig('/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_permodel.png',
                dpi=200, bbox_inches='tight')
    plt.close()

def main():
    print("Loading data...")
    dfs, min_date, max_date = load_data()
    print(f"Fit window: {min_date} to {max_date}")
    print()

    # Fit per-family models
    print("Fitting per-family conjugate models...")
    posteriors = {}
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        y = dfs[family]['cost'].values
        try:
            posterior = fit_conjugate_family(y)
            posteriors[family] = posterior
            print(f"{family}:")
            print(f"  Active days: {posterior['n_active']}")
            print(f"  Total hist: ${posterior['total_hist']:.2f}")
            print(f"  p_post: Beta{posterior['p_post']} (mean={posterior['p_post'][0]/(posterior['p_post'][0]+posterior['p_post'][1]):.3f})")
            print(f"  mu_n={posterior['mu_n']:.3f}, sigma_est={np.sqrt(posterior['beta_n']/posterior['alpha_n']):.3f}")
            if posterior['low_confidence']:
                print(f"  ** LOW CONFIDENCE (n_active < 3) **")
            print()
        except ValueError as e:
            print(f"{family}: SKIPPED - {e}")
            posteriors[family] = None
    print()

    # MC forecast per family
    print("Forecasting 30 days ahead (MC, n=6000)...")
    np.random.seed(42)
    M_dict = {}
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        if posteriors[family] is None:
            continue
        M_fam = forecast_30days_per_family_mc(posteriors[family], max_date, n_draws=6000)
        M_dict[family] = M_fam
    print()

    # Aggregate forecast
    n_draws = 6000
    M_agg = np.zeros((n_draws, 30))
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        if family in M_dict:
            M_agg += M_dict[family]

    # Per-model 30-day totals
    print("Per-model 30-day forecasts:")
    print(f"{'Family':<12} {'30d Median':<15} {'90% CI':<30}")
    print("-" * 60)
    for family in ['Opus', 'Sonnet', 'Haiku', 'Fable']:
        if family not in M_dict:
            print(f"{family:<12} {'N/A':<15}")
            continue
        M_fam = M_dict[family]
        totals = M_fam.sum(axis=1)
        med = np.median(totals)
        q5 = np.percentile(totals, 5)
        q95 = np.percentile(totals, 95)
        conf = " (low-conf)" if posteriors[family]['low_confidence'] else ""
        print(f"{family:<12} ${med:<14.0f} ${q5:.0f} - ${q95:.0f}{conf}")
    print()

    # Aggregate 30-day total
    agg_totals = M_agg.sum(axis=1)
    agg_med = np.median(agg_totals)
    agg_mean = np.mean(agg_totals)
    agg_q5 = np.percentile(agg_totals, 5)
    agg_q95 = np.percentile(agg_totals, 95)

    print("Aggregate 30-day forecast:")
    print(f"  Median: ${agg_med:.0f}")
    print(f"  Mean: ${agg_mean:.0f}")
    print(f"  90% CI: ${agg_q5:.0f} - ${agg_q95:.0f}")
    print()

    # Cross-check vs pooled
    pooled_ref = 254
    print(f"Cross-check vs pooled model:")
    print(f"  Pooled median: ${pooled_ref}")
    print(f"  Per-model aggregate median: ${agg_med:.0f}")
    if abs(agg_med - pooled_ref) < 20:
        print(f"  Status: CLOSE (diff = ${abs(agg_med - pooled_ref):.0f})")
    else:
        print(f"  Status: DIFFERENT (diff = ${abs(agg_med - pooled_ref):.0f})")
    print()

    # Create plot
    print("Creating forecast plot...")
    plot_permodel_forecast(dfs, posteriors, M_dict, max_date, min_date)
    png_path = '/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_permodel.png'
    print(f"Plot saved to {png_path}")

    # Report PNG stats
    if os.path.exists(png_path):
        stat = os.stat(png_path)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        size = stat.st_size
        print(f"  mtime: {mtime}")
        print(f"  size: {size} bytes")
    print()

    # Build aggregate sample (1000 draws from 6000 for dot grid)
    agg_totals_sample = list(np.percentile(agg_totals, np.linspace(0, 100, 1000)))

    # Compute per-family 30-day totals, CIs, and historical split
    families = ['Opus', 'Sonnet', 'Haiku', 'Fable']
    totals_30d = {}
    q5_dict = {}
    q95_dict = {}
    for family in families:
        M_fam = M_dict[family]
        totals_per_draw = M_fam.sum(axis=1)
        totals_30d[family] = np.median(totals_per_draw)
        q5_dict[family] = np.percentile(totals_per_draw, 5)
        q95_dict[family] = np.percentile(totals_per_draw, 95)

    hist_split = {}
    for family in families:
        hist_split[family] = float(posteriors[family]['total_hist'])

    # Build JSON output
    results_json = {
        'as_of': str(max_date),
        'fit_days': int((max_date - min_date).days + 1),
        'aggregate': {
            'median': float(np.round(agg_med, 2)),
            'mean': float(np.round(agg_mean, 2)),
            'ci5': float(np.round(agg_q5, 2)),
            'ci95': float(np.round(agg_q95, 2))
        },
        'pooled_ref_median': 254,
        'per_model': {
            'Opus': {
                'active_days': int(posteriors['Opus']['n_active']),
                'hist_total': float(np.round(posteriors['Opus']['total_hist'], 2)),
                'median30': float(np.round(totals_30d['Opus'], 2)),
                'ci5': float(np.round(q5_dict['Opus'], 2)),
                'ci95': float(np.round(q95_dict['Opus'], 2)),
                'p_active': float(posteriors['Opus']['p_post'][0] / sum(posteriors['Opus']['p_post']))
            },
            'Sonnet': {
                'active_days': int(posteriors['Sonnet']['n_active']),
                'hist_total': float(np.round(posteriors['Sonnet']['total_hist'], 2)),
                'median30': float(np.round(totals_30d['Sonnet'], 2)),
                'ci5': float(np.round(q5_dict['Sonnet'], 2)),
                'ci95': float(np.round(q95_dict['Sonnet'], 2)),
                'p_active': float(posteriors['Sonnet']['p_post'][0] / sum(posteriors['Sonnet']['p_post']))
            },
            'Haiku': {
                'active_days': int(posteriors['Haiku']['n_active']),
                'hist_total': float(np.round(posteriors['Haiku']['total_hist'], 2)),
                'median30': float(np.round(totals_30d['Haiku'], 2)),
                'ci5': float(np.round(q5_dict['Haiku'], 2)),
                'ci95': float(np.round(q95_dict['Haiku'], 2)),
                'p_active': float(posteriors['Haiku']['p_post'][0] / sum(posteriors['Haiku']['p_post']))
            },
            'Fable': {
                'active_days': int(posteriors['Fable']['n_active']),
                'hist_total': float(np.round(posteriors['Fable']['total_hist'], 2)),
                'median30': float(np.round(totals_30d['Fable'], 2)),
                'ci5': float(np.round(q5_dict['Fable'], 2)),
                'ci95': float(np.round(q95_dict['Fable'], 2)),
                'p_active': float(posteriors['Fable']['p_post'][0] / sum(posteriors['Fable']['p_post'])),
                'low_confidence': True
            }
        },
        'hist_split': {
            'Opus': float(np.round(hist_split['Opus'], 2)),
            'Sonnet': float(np.round(hist_split['Sonnet'], 2)),
            'Haiku': float(np.round(hist_split['Haiku'], 2)),
            'Fable': float(np.round(hist_split['Fable'], 2))
        },
        'agg_totals_sample': agg_totals_sample
    }

    json_path = '/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"JSON results saved to {json_path}")
    if os.path.exists(json_path):
        stat = os.stat(json_path)
        size = stat.st_size
        print(f"  size: {size} bytes")
    print()

if __name__ == '__main__':
    main()
