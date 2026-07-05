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

# Configuration
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

def load_data():
    """Load usage data from SQLite, compute daily costs, return DataFrame."""
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

    daily_costs = {}
    chicago_tz = ZoneInfo("America/Chicago")

    for timestamp_str, model, inp_tok, out_tok, cache_cre_tok, cache_read_tok in rows:
        # Skip if model is NULL or not priced
        if model is None:
            continue
        price = get_price(model)
        if price is None:
            continue

        # Parse timestamp: replace trailing Z with +00:00
        iso_str = timestamp_str.replace('Z', '+00:00')
        dt_utc = datetime.fromisoformat(iso_str)
        dt_chicago = dt_utc.astimezone(chicago_tz)
        date = dt_chicago.date()

        # Compute cost (USD)
        cost = (inp_tok * price['input'] +
                out_tok * price['output'] +
                cache_cre_tok * price['cache_write'] +
                cache_read_tok * price['cache_read']) / 1e6

        if date not in daily_costs:
            daily_costs[date] = 0.0
        daily_costs[date] += cost

    if not daily_costs:
        raise ValueError("No data loaded from database")

    # Build complete date range
    min_date = min(daily_costs.keys())
    max_date = max(daily_costs.keys())
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    data = []
    for d in date_range:
        date_obj = d.date()
        cost = daily_costs.get(date_obj, 0.0)
        weekday = d.weekday()  # Monday=0, Sunday=6
        data.append({'date': date_obj, 'cost': cost, 'weekday': weekday})

    df = pd.DataFrame(data)
    return df

def load_desktop_data():
    """Load usage data from desktop app local-agent-mode audit logs, return daily costs dict."""
    path_pattern = os.path.expanduser("~/Library/Application Support/Claude/local-agent-mode-sessions/**/audit.jsonl")
    audit_files = glob.glob(path_pattern, recursive=True)

    daily_costs = {}
    chicago_tz = ZoneInfo("America/Chicago")

    for audit_file in audit_files:
        try:
            with open(audit_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract fields
                    timestamp_str = record.get("_audit_timestamp")
                    message = record.get("message")

                    if not timestamp_str or not isinstance(message, dict):
                        continue

                    usage = message.get("usage")
                    if not usage or not isinstance(usage, dict):
                        continue

                    model = message.get("model")
                    if not model or model == "<synthetic>":
                        continue

                    price = get_price(model)
                    if price is None:
                        continue

                    # Parse timestamp: replace trailing Z with +00:00
                    iso_str = timestamp_str.replace('Z', '+00:00')
                    dt_utc = datetime.fromisoformat(iso_str)
                    dt_chicago = dt_utc.astimezone(chicago_tz)
                    date = dt_chicago.date()

                    # Extract token counts (default 0 if missing)
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    cache_creation_input_tokens = usage.get('cache_creation_input_tokens', 0)
                    cache_read_input_tokens = usage.get('cache_read_input_tokens', 0)

                    # Compute cost (USD)
                    cost = (input_tokens * price['input'] +
                            output_tokens * price['output'] +
                            cache_creation_input_tokens * price['cache_write'] +
                            cache_read_input_tokens * price['cache_read']) / 1e6

                    if date not in daily_costs:
                        daily_costs[date] = 0.0
                    daily_costs[date] += cost

        except (IOError, OSError):
            continue

    return daily_costs

def fit_conjugate_hurdle(y, w):
    """
    Fit closed-form conjugate Bayesian hurdle model.
    y: daily costs (N,)
    w: weekdays (N,) where 0-4 = weekday, 5-6 = weekend

    Returns: dict with posterior params
      - p_weekday_post: (alpha_post, beta_post) Beta posterior for weekday
      - p_weekend_post: (alpha_post, beta_post) Beta posterior for weekend
      - mu_post, kappa_post, alpha_post, beta_post: NIG posterior on log-costs
      - weekend_delta_shrunk: weekend mean shift (shrunk toward 0)
      - active_costs_log: log of active daily costs (for reference)
    """
    np.random.seed(42)

    # Split by group
    is_weekday = w < 5
    is_weekend = w >= 5

    n_weekday = np.sum(is_weekday)
    n_weekend = np.sum(is_weekend)

    k_weekday = np.sum((y > 0) & is_weekday)
    k_weekend = np.sum((y > 0) & is_weekend)

    # Occurrence: Beta-Binomial conjugate (Beta prior (1,1) = Jeffreys)
    # Posterior Beta(1 + k, 1 + n - k)
    p_weekday_post = (1 + k_weekday, 1 + n_weekday - k_weekday)
    p_weekend_post = (1 + k_weekend, 1 + n_weekend - k_weekend)

    # Amount: Normal-Inverse-Gamma on log-costs (active days only)
    active_mask = y > 0
    active_costs = y[active_mask]
    active_w = w[active_mask]

    if len(active_costs) == 0:
        raise ValueError("No active days in data")

    log_costs = np.log(active_costs)

    # Split active by group
    active_weekday_mask = active_w < 5
    active_weekend_mask = active_w >= 5

    log_weekday = log_costs[active_weekday_mask]
    log_weekend = log_costs[active_weekend_mask]

    # Pooled NIG prior
    mu0 = np.mean(log_costs)
    kappa0 = 1.0
    alpha0 = 2.0
    beta0 = 0.5 * np.var(log_costs) * alpha0

    # Posterior with pooled data
    n = len(log_costs)
    mean_log = np.mean(log_costs)
    var_log = np.var(log_costs, ddof=1) if n > 1 else 0.1

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean_log) / kappa_n
    alpha_n = alpha0 + n / 2.0

    # Sum of squared deviations
    ssd = np.sum((log_costs - mean_log) ** 2)
    beta_n = beta0 + 0.5 * ssd + 0.5 * kappa0 * n / kappa_n * (mean_log - mu0) ** 2

    # Compute weekend mean shift with shrinkage
    if len(log_weekend) > 0 and len(log_weekday) > 0:
        raw_delta = np.mean(log_weekend) - np.mean(log_weekday)
        # Shrink by k_weekend / (k_weekend + 5)
        shrink_factor = len(log_weekend) / (len(log_weekend) + 5.0)
        weekend_delta_shrunk = raw_delta * shrink_factor
    else:
        weekend_delta_shrunk = 0.0

    return {
        'p_weekday_post': p_weekday_post,
        'p_weekend_post': p_weekend_post,
        'mu_n': mu_n,
        'kappa_n': kappa_n,
        'alpha_n': alpha_n,
        'beta_n': beta_n,
        'weekend_delta_shrunk': weekend_delta_shrunk,
        'active_costs_log': log_costs,
        'k_weekday': k_weekday,
        'n_weekday': n_weekday,
        'k_weekend': k_weekend,
        'n_weekend': n_weekend,
    }

def forecast_30days_mc(posterior, max_date, n_draws=4000):
    """
    Monte Carlo forecast 30 days ahead using posterior samples.
    posterior: dict from fit_conjugate_hurdle()
    max_date: last date in historical data
    n_draws: number of MC draws

    Returns: (forecast_df, total_30d_mean, total_30d_5, total_30d_95)
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

    # MC: sample posterior params
    forecast_costs = {i: [] for i in range(horizon)}
    forecast_exp = {i: [] for i in range(horizon)}

    for draw_idx in range(n_draws):
        # Sample p_weekday, p_weekend from Beta posteriors
        p_wd = np.random.beta(alpha_wd, beta_wd)
        p_we = np.random.beta(alpha_we, beta_we)

        # Sample (sigma^2, mu) from NIG posterior
        # sigma^2 ~ InvGamma(alpha_n, beta_n)
        sigma2 = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
        sigma = np.sqrt(sigma2)

        # mu | sigma^2 ~ Normal(mu_n, sigma^2 / kappa_n)
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

            # Analytic expected value E[y|w] = p_group * exp(mu + group_shift + sigma^2/2)
            exp_val = p_group * np.exp(mu + group_shift + 0.5 * sigma2)
            forecast_exp[day_idx].append(exp_val)

            # Sample occurrence z ~ Bernoulli(p_group)
            z = np.random.binomial(1, p_group)

            if z == 1:
                # Sample log_y ~ Normal(mu + group_shift, sigma^2)
                log_y = np.random.normal(mu + group_shift, sigma)
                y = np.exp(log_y)
            else:
                y = 0.0

            forecast_costs[day_idx].append(y)

    # Compute quantiles
    forecast_data = []
    for day_idx, d in enumerate(forecast_dates):
        costs = np.array(forecast_costs[day_idx])
        mean_cost = costs.mean()
        q5 = np.percentile(costs, 5)
        q25 = np.percentile(costs, 25)
        q50 = np.percentile(costs, 50)
        q75 = np.percentile(costs, 75)
        q95 = np.percentile(costs, 95)
        exp_mean = np.mean(forecast_exp[day_idx])

        forecast_data.append({
            'date': d.date(),
            'weekday': d.weekday(),
            'weekday_name': d.strftime('%a'),
            'mean': mean_cost,
            'exp_mean': exp_mean,
            'q5': q5,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'q95': q95,
        })

    forecast_df = pd.DataFrame(forecast_data)

    # 30-day totals
    total_30d_draws = np.sum([np.array(forecast_costs[i]) for i in range(horizon)], axis=0)
    total_30d_mean = total_30d_draws.mean()
    total_30d_5 = np.percentile(total_30d_draws, 5)
    total_30d_95 = np.percentile(total_30d_draws, 95)

    return forecast_df, total_30d_mean, total_30d_5, total_30d_95

def compute_weekday_effects(posterior, n_draws=4000):
    """
    Compute per-weekday expected daily cost E[y|w] from posterior.
    Returns: (weekday_names, means, q5s, q95s)
    """
    np.random.seed(42)

    alpha_wd, beta_wd = posterior['p_weekday_post']
    alpha_we, beta_we = posterior['p_weekend_post']

    mu_n = posterior['mu_n']
    kappa_n = posterior['kappa_n']
    alpha_n = posterior['alpha_n']
    beta_n = posterior['beta_n']
    delta = posterior['weekend_delta_shrunk']

    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_costs = {w: [] for w in range(7)}

    for draw_idx in range(n_draws):
        p_wd = np.random.beta(alpha_wd, beta_wd)
        p_we = np.random.beta(alpha_we, beta_we)

        sigma2 = 1.0 / np.random.gamma(alpha_n, 1.0 / beta_n)
        mu = np.random.normal(mu_n, np.sqrt(sigma2 / kappa_n))

        for w in range(7):
            if w < 5:  # Weekday
                p_w = p_wd
                group_shift = -0.5 * delta
            else:  # Weekend
                p_w = p_we
                group_shift = 0.5 * delta

            expected_cost = p_w * np.exp(mu + group_shift + 0.5 * sigma2)
            weekday_costs[w].append(expected_cost)

    means = []
    q5s = []
    q95s = []
    for w in range(7):
        costs = np.array(weekday_costs[w])
        means.append(costs.mean())
        q5s.append(np.percentile(costs, 5))
        q95s.append(np.percentile(costs, 95))

    return weekday_names, means, q5s, q95s

def plot_forecast(df_combined, forecast_df, posterior, total_30d_mean, total_30d_5, total_30d_95):
    """Create and save forecast plot."""
    plt.style.use('seaborn-v0_8-darkgrid')

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # Left panel: historical + forecast
    hist_dates = df_combined['date'].values
    cli_costs = df_combined['cli_cost'].values
    desktop_costs = df_combined['desktop_cost'].values

    ax_left.bar(hist_dates, cli_costs, color='#4C72B0', alpha=0.85, width=0.8, label='Claude Code (CLI)')
    ax_left.bar(hist_dates, desktop_costs, bottom=cli_costs, color='#55A868', alpha=0.85, width=0.8, label='Desktop app (local agent)')

    # Boundary line
    boundary = hist_dates[-1]
    ax_left.axvline(boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Forecast
    forecast_dates = forecast_df['date'].values
    q5 = forecast_df['q5'].values
    q25 = forecast_df['q25'].values
    q50 = forecast_df['q50'].values
    q75 = forecast_df['q75'].values
    q95 = forecast_df['q95'].values
    exp_mean = forecast_df['exp_mean'].values

    ax_left.fill_between(forecast_dates, q5, q95, color='#DD8452', alpha=0.20, label='90% interval')
    ax_left.fill_between(forecast_dates, q25, q75, color='#DD8452', alpha=0.35, label='50% interval')
    ax_left.plot(forecast_dates, q50, color='#DD8452', marker='o', markersize=3, linewidth=1.5, label='median')
    ax_left.plot(forecast_dates, exp_mean, color='#C44E52', linestyle='--', linewidth=1.5, label='expected value')

    ax_left.set_title('Claude usage — conjugate hurdle model (weekend contrast)', fontsize=11)
    ax_left.set_ylabel('USD/day')
    ax_left.set_xlabel('Date')
    ax_left.legend(loc='upper left', fontsize=9)

    ax_left.tick_params(axis='x', rotation=45)
    fig.autofmt_xdate()

    # Annotation
    textstr = f"Next 30d: ${total_30d_mean:.0f} (90% CI ${total_30d_5:.0f}–${total_30d_95:.0f})"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_left.text(0.02, 0.70, textstr, transform=ax_left.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Context note
    history_note = "History: desktop app (Apr) + CLI (Jun+). May = no surviving data.\nForecast = CLI, closed-form conjugate (no MCMC)."
    props_note = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_left.text(0.02, 0.57, history_note, transform=ax_left.transAxes, fontsize=8,
                verticalalignment='top', bbox=props_note)

    # Right panel: weekday effects
    weekday_names, means, q5s, q95s = compute_weekday_effects(posterior)
    errors = np.array([np.array(means) - np.array(q5s), np.array(q95s) - np.array(means)])

    ax_right.bar(range(7), means, color='#4C72B0', alpha=0.85)
    ax_right.errorbar(range(7), means, yerr=errors, fmt='none', ecolor='black', capsize=3, alpha=0.6)
    ax_right.set_xticks(range(7))
    ax_right.set_xticklabels(weekday_names, rotation=45, ha='right', fontsize=9)
    ax_right.set_title('Weekday effect (expected $/day)', fontsize=11)
    ax_right.set_ylabel('Expected cost (USD)')

    plt.savefig('/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    desktop = load_desktop_data()

    # Fit window: CLI daily series from first to last CLI date
    cli_dates = df['date'].values
    min_cli_date = df['date'].min()
    max_cli_date = df['date'].max()

    min_history_date = min_cli_date
    if desktop:
        min_history_date = min(min_history_date, min(desktop.keys()))

    # Fit window: CLI only
    y = df['cost'].values
    w = df['weekday'].values

    n_days = len(y)
    n_positive = np.sum(y > 0)

    print(f"Fit window (CLI only): {n_days} days ({min_cli_date} to {max_cli_date})")
    print(f"  Active days: {n_positive} ({100*n_positive/n_days:.1f}%)")
    print()

    # Fit conjugate model
    print("Fitting conjugate Bayesian hurdle model...")
    posterior = fit_conjugate_hurdle(y, w)
    print(f"  Weekday occurrence: p ~ Beta{posterior['p_weekday_post']}")
    print(f"  Weekend occurrence: p ~ Beta{posterior['p_weekend_post']}")
    print(f"  Log-amount NIG: mu_post={posterior['mu_n']:.3f}, sigma_post^2~InvGamma({posterior['alpha_n']:.1f}, {posterior['beta_n']:.3f})")
    print(f"  Weekend mean shift (shrunk): {posterior['weekend_delta_shrunk']:.3f}")
    print()

    # Forecast
    print("Forecasting 30 days ahead (MC, n=4000)...")
    forecast_df, total_30d_mean, total_30d_5, total_30d_95 = forecast_30days_mc(posterior, max_cli_date, n_draws=4000)
    print(f"  30-day total: ${total_30d_mean:.0f} (90% CI ${total_30d_5:.0f}–${total_30d_95:.0f})")
    print()

    # Weekday table
    weekday_names, weekday_means, _, _ = compute_weekday_effects(posterior)
    print("Per-weekday expected daily cost (posterior mean):")
    print(f"{'Weekday':<12} {'E[cost|w]':<12}")
    print("-" * 24)
    for w in range(7):
        print(f"{weekday_names[w]:<12} ${weekday_means[w]:<11.2f}")
    print()

    # 30-day forecast table
    print("30-day forecast:")
    print(f"{'Date':<12} {'Day':<12} {'Mean':<10} {'E[y]':<10} {'Q5':<10} {'Q50':<10} {'Q95':<10}")
    print("-" * 75)
    for _, row in forecast_df.iterrows():
        print(f"{str(row['date']):<12} {row['weekday_name']:<12} ${row['mean']:<9.2f} ${row['exp_mean']:<9.2f} ${row['q5']:<9.2f} ${row['q50']:<9.2f} ${row['q95']:<9.2f}")
    print()

    # Build combined history for plotting
    full_date_range = pd.date_range(start=min_history_date, end=max_cli_date, freq='D')
    combined_data = []
    for d in full_date_range:
        date_obj = d.date()
        cli_cost = float(df[df['date'] == date_obj]['cost'].values[0]) if date_obj in df['date'].values else 0.0
        desktop_cost = desktop.get(date_obj, 0.0)
        combined_data.append({'date': date_obj, 'cli_cost': cli_cost, 'desktop_cost': desktop_cost, 'weekday': d.weekday()})

    df_combined = pd.DataFrame(combined_data)

    # Plot
    print("Creating forecast plot...")
    plot_forecast(df_combined, forecast_df, posterior, total_30d_mean, total_30d_5, total_30d_95)
    print(f"Plot saved to /private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_forecast.png")
    print()

if __name__ == '__main__':
    main()
