#!/usr/bin/env python3
"""
FiveThirtyEight-style forecast card for Claude API spend.
Reads results.json and creates a clean, layperson-friendly editorial visualization.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

def make_card():
    results = load_results('/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/results.json')

    # Extract values
    agg_med = results['aggregate']['median']
    agg_mean = results['aggregate']['mean']
    agg_ci5 = results['aggregate']['ci5']
    agg_ci95 = results['aggregate']['ci95']

    hist_split = results['hist_split']
    as_of = results['as_of']

    # Sample for dot grid
    agg_sample = results['agg_totals_sample']

    # Figure setup - stacked GridSpec to prevent overlap
    fig = plt.figure(figsize=(10, 15), facecolor='white')
    fig.patch.set_facecolor('white')
    gs = GridSpec(6, 1, figure=fig, height_ratios=[1.1, 1.5, 2.0, 4.2, 2.2, 0.5], hspace=0.55)

    # ===== ZONE 1: HEADLINE =====
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')

    headline_text = "How much will Claude\ncost you next month?"
    ax_title.text(0.5, 0.5, headline_text, fontsize=24, fontweight='bold',
                  ha='center', va='center', family='sans-serif', color='#333333')
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)

    # ===== ZONE 2: BIG NUMBER + SUBTITLE =====
    ax_number = fig.add_subplot(gs[1])
    ax_number.axis('off')

    # Big number with escaped dollar sign
    big_num = f"\\${agg_med:.0f}"
    ax_number.text(0.5, 0.62, big_num, fontsize=52, fontweight='bold',
                   ha='center', va='center', family='sans-serif', color='#1E6FBA',
                   transform=ax_number.transAxes)

    # Subtitle low in this zone
    subtitle_text = "your most likely 30-day spend"
    ax_number.text(0.5, 0.10, subtitle_text, fontsize=13, ha='center', va='bottom',
                   family='sans-serif', color='#666666', style='italic', transform=ax_number.transAxes)
    ax_number.set_xlim(0, 1)
    ax_number.set_ylim(0, 1)

    # ===== ZONE 3: RANGE BAR =====
    ax_range = fig.add_subplot(gs[2])
    ax_range.set_xlim(agg_ci5 * 0.7, agg_ci95 * 1.1)
    ax_range.set_ylim(0, 1)
    ax_range.axis('off')

    # Draw the outcome range bar
    bar_y = 0.5
    bar_height = 0.08

    ax_range.barh(bar_y, agg_ci95 - agg_ci5, left=agg_ci5, height=bar_height,
                  color='#1E6FBA', alpha=0.4, edgecolor='#1E6FBA', linewidth=2)

    # Median marker
    ax_range.plot([agg_med, agg_med], [bar_y - bar_height/2 - 0.08, bar_y + bar_height/2 + 0.08],
                  color='#FF4136', linewidth=4, solid_capstyle='round')

    # Labels: typical above bar, light/heavy below
    ax_range.text(agg_med, 0.8, f'Typical\n\\${agg_med:.0f}', fontsize=10,
                  ha='center', va='bottom', family='sans-serif', color='#FF4136', fontweight='bold')

    ax_range.text(agg_ci5, 0.15, f'Light month\n~\\${agg_ci5:.0f}', fontsize=10,
                  ha='center', va='top', family='sans-serif', color='#666666', fontweight='bold')

    ax_range.text(agg_ci95, 0.15, f'Heavy month\n~\\${agg_ci95:.0f}', fontsize=10,
                  ha='center', va='top', family='sans-serif', color='#666666', fontweight='bold')

    # Sentence at the top of ax_dots instead (to avoid overlap here)

    # ===== ZONE 4: 100 POSSIBLE MONTHS DOT GRID =====
    ax_dots = fig.add_subplot(gs[3])
    ax_dots.axis('off')
    ax_dots.set_xlim(-0.5, 10.5)
    ax_dots.set_ylim(-0.5, 10.5)

    # "Each dot" label at top of dot grid
    ax_dots.text(0.5, 0.92, 'Each dot = 1 in 100 possible months', fontsize=10, ha='center', va='top',
                 family='sans-serif', color='#666666', transform=ax_dots.transAxes, fontweight='bold')

    # Create 100 dots from the sample
    dot_sample = [agg_sample[i] for i in range(0, len(agg_sample), len(agg_sample) // 100)][:100]
    dot_sample = sorted(dot_sample)

    # Categorize dots
    colors_dots = []
    for val in dot_sample:
        if val < 150:
            colors_dots.append('#CCCCCC')  # Light gray
        elif val <= 400:
            colors_dots.append('#1E6FBA')  # Accent blue
        else:
            colors_dots.append('#333333')  # Dark

    # Draw 10x10 grid (centered in the 0-10 space)
    for i, val in enumerate(dot_sample):
        row = i // 10
        col = i % 10
        ax_dots.scatter(col, 9 - row, s=200, c=colors_dots[i], alpha=0.8, edgecolors='white', linewidth=1)

    # Bucket legend at the BOTTOM of this zone (doesn't overlap grid or money section)
    leg_handles = [
        plt.scatter([], [], s=120, c='#CCCCCC', edgecolors='white', linewidth=1),
        plt.scatter([], [], s=120, c='#1E6FBA', edgecolors='white', linewidth=1),
        plt.scatter([], [], s=120, c='#333333', edgecolors='white', linewidth=1)
    ]
    leg_labels = ['Under \\$150', '\\$150-400', 'Over \\$400']
    ax_dots.legend(leg_handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, 0.04),
                   ncol=3, frameon=False, fontsize=9, handletextpad=0.6)

    # ===== ZONE 5: WHERE THE MONEY GOES =====
    ax_money = fig.add_subplot(gs[4])
    ax_money.axis('off')

    # Heading
    ax_money.text(0.5, 0.92, 'Where the money comes from (last month)', fontsize=12,
                  ha='center', va='top', family='sans-serif', color='#333333', fontweight='bold',
                  transform=ax_money.transAxes)

    # Calculate totals and percentages
    total_hist = sum(hist_split.values())
    models_order = ['Opus', 'Sonnet', 'Haiku', 'Fable']
    model_colors = {'Opus': '#4C72B0', 'Sonnet': '#55A868', 'Haiku': '#DD8452', 'Fable': '#C44E52'}

    # Money split rows - evenly spaced
    y_positions = [0.72, 0.56, 0.40, 0.24]
    for model, y_pos in zip(models_order, y_positions):
        hist_val = hist_split[model]
        pct = 100 * hist_val / total_hist
        # Escaped dollar sign in f-string
        label = f"{model:8s}  \\${hist_val:6.0f}  ({pct:5.1f}%)"
        ax_money.text(0.05, y_pos, label, fontsize=10, family='monospace', va='top',
                     color='#333333', fontweight='bold', transform=ax_money.transAxes)

    # Fable callout at the very bottom
    fable_text = "Fable costs 2x per word - 2 days of it = \\$27, about the same as 5 weeks of Sonnet."
    ax_money.text(0.05, 0.05, fable_text, fontsize=9, family='sans-serif', va='top',
                 color='#666666', style='italic', transform=ax_money.transAxes)

    ax_money.set_xlim(0, 1)
    ax_money.set_ylim(0, 1)

    # ===== ZONE 6: CAPTION (via fig.text) =====
    caption = "Based on about 5 weeks of usage (Jun 2 - Jul 5). Still early - the range will narrow as more data comes in. Source: local Claude usage logs."
    fig.text(0.5, 0.015, caption, ha='center', fontsize=9, color='#999999', style='italic')

    plt.savefig('/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_forecast_card.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print("FiveThirtyEight-style forecast card created.")
    png_path = '/private/tmp/claude-501/-Users-domattioli-Projects-DomI/da8d7db4-b687-449f-b58e-10be8c49646f/scratchpad/usage_forecast_card.png'
    import os
    if os.path.exists(png_path):
        stat = os.stat(png_path)
        size = stat.st_size
        from datetime import datetime
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        print(f"PNG saved: {png_path}")
        print(f"  mtime: {mtime}")
        print(f"  size: {size} bytes")

if __name__ == '__main__':
    make_card()
