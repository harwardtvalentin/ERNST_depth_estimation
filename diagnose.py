"""
Diagnostic script for depth estimation analysis - all methods including iterative.
Run from project root: python diagnose.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Load CSV: automatically find most recent run ──────────────────────────────
runs_dir = 'experiments/runs'
run_folders = sorted([
    f for f in os.listdir(runs_dir)
    if f.startswith('single_') and os.path.isdir(os.path.join(runs_dir, f))
])
latest_run = run_folders[-1]
RUN_DIR   = os.path.join(runs_dir, latest_run)
CSV_PATH  = os.path.join(RUN_DIR, 'depth_estimates.csv')
LOG_PATH  = os.path.join(RUN_DIR, 'diagnose_output.txt')

# ── Tee stdout to both terminal and log file ──────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_log_file = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _log_file)

print(f"Loading: {CSV_PATH}")
print(f"Log:     {LOG_PATH}\n")

df = pd.read_csv(CSV_PATH, sep=';')

# ── Column maps ───────────────────────────────────────────────────────────────
METHODS = {
    'two_ray':       'depth_estimate_two_ray',
    'multi_ray':     'depth_estimate_multi_ray',
    'kalman':        'depth_estimate_kalman',
    'iterative_k1':  'depth_estimate_iterative_k1',
    'iterative_k2':  'depth_estimate_iterative_k2',
    'iterative_k3':  'depth_estimate_iterative_k3',
    'iterative_k4':  'depth_estimate_iterative_k4',
    'iterative_k5':  'depth_estimate_iterative_k5',
}
ERROR_COLS = {
    'two_ray':       'depth_error_two_ray',
    'multi_ray':     'depth_error_multi_ray',
    'kalman':        'depth_error_kalman',
    'iterative_k1':  'depth_error_iterative_k1',
    'iterative_k2':  'depth_error_iterative_k2',
    'iterative_k3':  'depth_error_iterative_k3',
    'iterative_k4':  'depth_error_iterative_k4',
    'iterative_k5':  'depth_error_iterative_k5',
}
GAP_COLS = {
    'two_ray':       'triangulation_gap_two_ray',
    'multi_ray':     'triangulation_gap_multi_ray',
    'kalman':        None,
    'iterative_k1':  None,
    'iterative_k2':  None,
    'iterative_k3':  None,
    'iterative_k4':  None,
    'iterative_k5':  None,
}
OFFSET_COLS = {
    'two_ray':       'time_offset_two_ray',
    'multi_ray':     'time_offset_multi_ray',
    'kalman':        None,
    'iterative_k1':  None,
    'iterative_k2':  None,
    'iterative_k3':  None,
    'iterative_k4':  None,
    'iterative_k5':  None,
}

ITERATIVE_KEYS = ['iterative_k1', 'iterative_k2', 'iterative_k3', 'iterative_k4', 'iterative_k5']
BASELINE_METHODS = ['two_ray', 'multi_ray', 'kalman']


def _rmse(errors):
    return np.sqrt((errors ** 2).mean())

def _col_present(method):
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    return est_col in df.columns and err_col in df.columns

def _subset(method):
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    return df[df[est_col].notna()].copy()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1 — OVERVIEW")
print("=" * 60)
print(f"Total rows:      {len(df)}")
print(f"Min true_depth:  {df['true_depth'].min()/1e3:.1f} km")
print(f"Max true_depth:  {df['true_depth'].max()/1e3:.1f} km\n")

for method, est_col in METHODS.items():
    err_col = ERROR_COLS[method]
    if not _col_present(method):
        print(f"{method.upper()}: column not found")
        continue
    subset = _subset(method)
    if len(subset) == 0:
        print(f"{method.upper()}: no estimates")
        continue
    errors = subset[err_col]
    print(f"{method.upper()}: {len(subset)} estimates")
    print(f"  RMSE:    {_rmse(errors)/1e3:8.1f} km")
    print(f"  Mean:    {errors.mean()/1e3:8.1f} km")
    print(f"  Median:  {errors.median()/1e3:8.1f} km")
    print(f"  Std:     {errors.std()/1e3:8.1f} km")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RMSE BY DISTANCE BIN
# ═════════════════════════════════════════════════════════════════════════════
bins = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]

# Show all baseline methods + iterative_k5 as the iterative representative
methods_for_bins = BASELINE_METHODS + ['iterative_k5']

for method in methods_for_bins:
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    if not _col_present(method):
        continue
    subset = _subset(method)
    if len(subset) == 0:
        continue

    print("=" * 60)
    print(f"SECTION 2 — {method.upper()}: RMSE by distance bin")
    print("=" * 60)
    for lo, hi in bins:
        s = subset[
            (subset['true_depth'] >= lo * 1e3) &
            (subset['true_depth'] <  hi * 1e3)
        ]
        if len(s) == 0:
            continue
        print(f"  {lo:5d}-{hi:5d} km:  "
              f"RMSE={_rmse(s[err_col])/1e3:7.1f} km   "
              f"mean={s[err_col].mean()/1e3:+7.1f} km   "
              f"({len(s):5d} estimates)")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RMSE BY TIME OFFSET (two_ray and multi_ray)
# ═════════════════════════════════════════════════════════════════════════════
for method in ['two_ray', 'multi_ray']:
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    off_col = OFFSET_COLS[method]
    gap_col = GAP_COLS[method]

    if not _col_present(method) or off_col not in df.columns:
        continue
    subset = _subset(method)
    if len(subset) == 0:
        continue

    print("=" * 60)
    print(f"SECTION 3 — {method.upper()}: RMSE by time offset")
    print("=" * 60)
    for offset in sorted(subset[off_col].dropna().unique()):
        s = subset[subset[off_col] == offset]
        if len(s) == 0:
            continue
        gap_str = ''
        if gap_col and gap_col in df.columns:
            gap_str = f"   gap={s[gap_col].median():7.0f} m"
        print(f"  Dt={offset:4.0f}s:  RMSE={_rmse(s[err_col])/1e3:7.1f} km{gap_str}   ({len(s):5d} estimates)")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ITERATIVE METHOD: CONVERGENCE ACROSS k
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 4 — ITERATIVE METHOD: CONVERGENCE ACROSS k=1..5")
print("=" * 60)

# Compute RMSE for k=1 as baseline for improvement percentage
k1_rmse = None
if _col_present('iterative_k1'):
    s = _subset('iterative_k1')
    if len(s) > 0:
        k1_rmse = _rmse(s[ERROR_COLS['iterative_k1']])

print(f"  {'k':<5} {'N':>7} {'RMSE (km)':>12} {'Mean (km)':>12} {'vs k=1':>10}")
print(f"  {'-'*5} {'-'*7} {'-'*12} {'-'*12} {'-'*10}")
for key in ITERATIVE_KEYS:
    if not _col_present(key):
        continue
    subset = _subset(key)
    if len(subset) == 0:
        continue
    err_col = ERROR_COLS[key]
    errors = subset[err_col]
    rmse = _rmse(errors)
    k = int(key[-1])
    vs_k1 = ''
    if k1_rmse is not None and k > 1:
        pct = 100.0 * (rmse - k1_rmse) / k1_rmse
        vs_k1 = f"{pct:+.1f}%"
    print(f"  {k:<5} {len(subset):>7} {rmse/1e3:>12.2f} {errors.mean()/1e3:>12.2f} {vs_k1:>10}")

# Sanity check: iterative_k1 vs two_ray at Dt=1s
print()
print("  Sanity check — iterative_k1 should match two_ray at Dt=1s:")
if _col_present('iterative_k1') and _col_present('two_ray') and OFFSET_COLS['two_ray'] in df.columns:
    k1 = _subset('iterative_k1')
    two_ray_1s = _subset('two_ray')
    two_ray_1s = two_ray_1s[two_ray_1s[OFFSET_COLS['two_ray']] == 1.0]
    if len(k1) > 0 and len(two_ray_1s) > 0:
        print(f"    iterative_k1 RMSE:     {_rmse(k1[ERROR_COLS['iterative_k1']])/1e3:.2f} km  ({len(k1)} estimates)")
        print(f"    two_ray Dt=1s RMSE:    {_rmse(two_ray_1s[ERROR_COLS['two_ray']])/1e3:.2f} km  ({len(two_ray_1s)} estimates)")
print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BIAS ANALYSIS (signed error) FOR ALL METHODS
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 5 — BIAS ANALYSIS (signed error, positive = overestimate)")
print("=" * 60)

print(f"  {'Method':<20} {'N':>7} {'Mean err (km)':>14} {'Median err (km)':>16} {'Std (km)':>10} {'Bias?':>12}")
print(f"  {'-'*20} {'-'*7} {'-'*14} {'-'*16} {'-'*10} {'-'*12}")

for method in list(METHODS.keys()):
    if not _col_present(method):
        continue
    subset = _subset(method)
    if len(subset) == 0:
        continue
    err_col = ERROR_COLS[method]
    errors = subset[err_col]
    mean_e = errors.mean()
    med_e  = errors.median()
    std_e  = errors.std()

    # Simple bias label
    if abs(mean_e) < 0.1 * std_e:
        bias_label = 'unbiased'
    elif mean_e > 0:
        bias_label = 'overestimate'
    else:
        bias_label = 'underestimate'

    print(f"  {method:<20} {len(subset):>7} {mean_e/1e3:>+14.2f} {med_e/1e3:>+16.2f} {std_e/1e3:>10.2f} {bias_label:>12}")
print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLOSEST APPROACH REGION (true_depth < 300 km)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 6 — CLOSEST APPROACH REGION (true_depth < 300 km)")
print("=" * 60)

for method, est_col in METHODS.items():
    err_col = ERROR_COLS[method]
    off_col = OFFSET_COLS[method]
    if not _col_present(method):
        continue
    subset = _subset(method)
    near = subset[subset['true_depth'] < 300e3]
    if len(near) == 0:
        print(f"\n{method.upper()}: no estimates near closest approach")
        continue

    errors = near[err_col]
    print(f"\n{method.upper()}: {len(near)} estimates")
    print(f"  RMSE:       {_rmse(errors)/1e3:.2f} km")
    print(f"  Mean error: {errors.mean()/1e3:+.2f} km")
    print(f"  Median err: {errors.median()/1e3:+.2f} km")
    print(f"  Std:        {errors.std()/1e3:.2f} km")

    if off_col and off_col in df.columns:
        print(f"  RMSE by offset:")
        for offset in sorted(near[off_col].dropna().unique()):
            s = near[near[off_col] == offset]
            if len(s) > 0:
                print(f"    Dt={offset:4.0f}s:  RMSE={_rmse(s[err_col])/1e3:.2f} km  ({len(s)} estimates)")
print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — AVAILABILITY AND LATENCY COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 7 — AVAILABILITY AND LATENCY COMPARISON")
print("=" * 60)

total_rows = len(df)

print(f"  {'Method':<20} {'N estimates':>12} {'Coverage':>10} {'First estimate (row)':>22} {'Last estimate (row)':>21}")
print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*22} {'-'*21}")

for method, est_col in METHODS.items():
    if not _col_present(method):
        continue
    subset = df[df[est_col].notna()]
    if len(subset) == 0:
        print(f"  {method:<20} {'0':>12} {'0.0%':>10}")
        continue
    coverage = 100.0 * len(subset) / total_rows
    first_row = subset.index[0]
    last_row  = subset.index[-1]
    print(f"  {method:<20} {len(subset):>12} {coverage:>9.1f}% {first_row:>22} {last_row:>21}")

# Latency note for iterative
print()
print("  Note: iterative methods require long_window + short_window ≈ 11s of history")
print("        before producing their first estimate (by design).")
print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRIANGULATION GAP VS ERROR CORRELATION
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 8 — TRIANGULATION GAP vs ERROR CORRELATION")
print("=" * 60)
print("  (Tests whether gap magnitude is a reliable quality predictor)")
print()

gap_bins = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000)]

for method in ['two_ray', 'multi_ray']:
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    gap_col = GAP_COLS[method]

    if not _col_present(method) or not gap_col or gap_col not in df.columns:
        continue
    subset = _subset(method)
    subset = subset[subset[gap_col].notna()]
    if len(subset) == 0:
        continue

    print(f"  {method.upper()}:")
    print(f"    {'Gap bin (m)':<22} {'N':>7} {'RMSE (km)':>12} {'Mean err (km)':>14}")
    print(f"    {'-'*22} {'-'*7} {'-'*12} {'-'*14}")
    for lo, hi in gap_bins:
        s = subset[
            (subset[gap_col] >= lo) &
            (subset[gap_col] <  hi)
        ]
        if len(s) == 0:
            continue
        print(f"    {lo:6d} – {hi:6d} m:   {len(s):>7} {_rmse(s[err_col])/1e3:>12.2f} {s[err_col].mean()/1e3:>+14.2f}")

    # Pearson correlation between gap and absolute error
    corr = subset[gap_col].corr(subset[err_col].abs())
    print(f"    Pearson correlation (gap vs |error|): {corr:.3f}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WORST AND BEST ESTIMATES PER METHOD
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 9 — WORST AND BEST ESTIMATES PER METHOD")
print("=" * 60)

# Show baseline methods + iterative_k5 as representative
methods_for_extremes = BASELINE_METHODS + ['iterative_k5']

for method in methods_for_extremes:
    est_col = METHODS[method]
    err_col = ERROR_COLS[method]
    off_col = OFFSET_COLS[method]
    gap_col = GAP_COLS[method]
    if not _col_present(method):
        continue
    subset = _subset(method)
    if len(subset) == 0:
        continue

    print(f"\n{method.upper()}:")
    for label, idx in [('Worst', subset[err_col].abs().idxmax()),
                        ('Best',  subset[err_col].abs().idxmin())]:
        row = subset.loc[idx]
        print(f"  {label}:")
        print(f"    true_depth: {row['true_depth']/1e3:.1f} km")
        print(f"    estimated:  {row[est_col]/1e3:.1f} km")
        print(f"    error:      {row[err_col]/1e3:+.1f} km")
        if gap_col and gap_col in df.columns:
            print(f"    gap:        {row[gap_col]:.0f} m")
        if off_col and off_col in df.columns and not pd.isna(row.get(off_col, float('nan'))):
            print(f"    offset:     {row[off_col]:.0f} s")
