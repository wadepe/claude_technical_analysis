"""
run_pipeline.py

Master orchestration script for the rising-wedge ML pipeline.
Runs the complete workflow for any bar-window size and stores all outputs
in an isolated run directory so results from different window sizes
never overwrite each other.

Pipeline steps
--------------
  1. Generate corpus        100K rising-wedge + 400K noise datasets
  2. Add BTC negatives      Real BTC windows as hard negatives
  3. Train 1D CNN           ~30 epochs, early stopping on val_auc
  4. Scan BTC               Roll a window across 2M 1-min BTC bars
  5. Backtest               Measure forward returns at 30m/1hr/2hr/4hr

Window-size scaling
-------------------
  All bar counts and trendline slopes scale proportionally so the rising-wedge
  geometry looks visually similar at every window size:
    250 bars  ~4.2 hrs   (baseline, full pipeline already run)
    100 bars  ~1.7 hrs
     50 bars  ~0.8 hrs
     30 bars  ~0.5 hrs

  The WEDGE_TOTAL_BARS environment variable is set for every subprocess so
  all scripts automatically use the correct window size without code changes.

Output layout
-------------
  project_root/
    runs/
      window_50bar/
        training_data/        corpus parquet files
        validation_data/
        numpy_cache/          preprocessed arrays (rebuilt if absent)
        corpus_manifest.json
        models/
          cnn_best.weights.h5
          cnn_final.weights.h5
          training_history.png
          training_log.csv
          evaluation/
          bitcoin_scan/
          backtest/

Usage
-----
  # Full pipeline for 50-bar windows
  python run_pipeline.py --window-size 50

  # Skip corpus (re-use existing) and jump straight to training
  python run_pipeline.py --window-size 100 --skip-corpus --skip-btc-neg

  # Run only the scan + backtest with an already-trained model
  python run_pipeline.py --window-size 50 --skip-corpus --skip-btc-neg --skip-train

  # Custom corpus sizes
  python run_pipeline.py --window-size 50 --n-wedge 50000 --n-noise 200000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR  = Path(__file__).parent


def _run(name: str, cmd: list[str], env: dict) -> None:
    """Run a subprocess step; abort the pipeline on failure."""
    print(f"\n{'='*64}")
    print(f"  STEP: {name}")
    print(f"{'='*64}")
    result = subprocess.run(cmd, env=env, cwd=str(SCRIPTS_DIR))
    if result.returncode != 0:
        print(f"\n[FAILED] {name}  (exit {result.returncode})")
        sys.exit(1)
    print(f"[DONE]  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rising-wedge ML pipeline — configurable bar window size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core setting
    parser.add_argument(
        '--window-size', type=int, default=250,
        help='Bars per dataset  (1 bar = 1 minute for BTC 1-min data, default: 250)',
    )

    # Corpus
    parser.add_argument('--n-wedge',    type=int,   default=100_000,
                        help='Rising-wedge datasets to generate (default: 100,000)')
    parser.add_argument('--n-noise',    type=int,   default=400_000,
                        help='Noise datasets to generate (default: 400,000)')
    parser.add_argument('--btc-stride', type=int,   default=None,
                        help='BTC-negative window stride '
                             '(default: 2 * window_size)')

    # Training
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=128)

    # Scan + backtest
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--scan-stride',type=int,   default=None,
                        help='Scan stride in bars  '
                             '(default: max(1, window_size // 25))')
    parser.add_argument('--cont-bars',  type=int,   default=None,
                        help='Bars of continuation for backtest charts  '
                             '(default: window_size)')

    # Skip flags
    parser.add_argument('--skip-corpus',   action='store_true',
                        help='Skip corpus generation (re-use existing)')
    parser.add_argument('--skip-btc-neg',  action='store_true',
                        help='Skip BTC-negative injection step')
    parser.add_argument('--skip-train',    action='store_true',
                        help='Skip model training')
    parser.add_argument('--skip-scan',     action='store_true',
                        help='Skip BTC scan')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtest')

    args = parser.parse_args()

    N           = args.window_size
    btc_stride  = args.btc_stride  or max(1, N * 2)
    scan_stride = args.scan_stride or max(1, N // 25)
    cont_bars   = args.cont_bars   or N

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = PROJECT_ROOT / 'runs' / f'window_{N}bar'
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path   = str(run_dir / 'models' / 'cnn_best.weights.h5')
    scan_out_dir   = str(run_dir / 'models' / 'bitcoin_scan')
    backtest_dir   = str(run_dir / 'models' / 'backtest')

    print(f"\n{'='*64}")
    print(f"  Rising-wedge Pipeline")
    print(f"{'='*64}")
    print(f"  Window size    : {N} bars  ({N} minutes of 1-min BTC data)")
    print(f"  Run directory  : {run_dir}")
    print(f"  BTC neg stride : {btc_stride} bars")
    print(f"  Scan stride    : {scan_stride} bars")
    print(f"  Cont. bars     : {cont_bars} bars")
    print(f"{'='*64}")

    # Propagate window size to all subprocesses via environment variable
    env = {
        **os.environ,
        'WEDGE_TOTAL_BARS':  str(N),
        'PYTHONIOENCODING':  'utf-8',
    }

    py = sys.executable   # use the same Python interpreter

    # ── Step 1: Corpus generation ─────────────────────────────────────────────
    if not args.skip_corpus:
        import shutil
        for d in ('training_data', 'validation_data', 'numpy_cache'):
            shutil.rmtree(run_dir / d, ignore_errors=True)
        manifest = run_dir / 'corpus_manifest.json'
        if manifest.exists():
            manifest.unlink()

        _run(
            f'1. Corpus generation  ({N}-bar windows)',
            [py, str(SCRIPTS_DIR / 'generate_rising_wedge.py'),
             '--corpus',
             '--n-wedge',    str(args.n_wedge),
             '--n-noise',    str(args.n_noise),
             '--output-dir', str(run_dir)],
            env,
        )

    # ── Step 2: BTC negatives ─────────────────────────────────────────────────
    if not args.skip_btc_neg:
        _run(
            f'2. BTC negatives  (stride={btc_stride})',
            [py, str(SCRIPTS_DIR / 'add_btc_negatives.py'),
             '--data-dir', str(run_dir),
             '--stride',   str(btc_stride)],
            env,
        )

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        _run(
            f'3. Train 1D CNN  ({args.epochs} max epochs)',
            [py, str(SCRIPTS_DIR / 'train_cnn.py'),
             '--data-dir',   str(run_dir),
             '--epochs',     str(args.epochs),
             '--batch-size', str(args.batch_size)],
            env,
        )

    # ── Step 4: BTC scan ──────────────────────────────────────────────────────
    if not args.skip_scan:
        _run(
            f'4. BTC scan  (stride={scan_stride}, threshold={args.threshold})',
            [py, str(SCRIPTS_DIR / 'scan_bitcoin.py'),
             '--stride',     str(scan_stride),
             '--threshold',  str(args.threshold),
             '--weights',    weights_path,
             '--output-dir', scan_out_dir],
            env,
        )

    # ── Step 5: Backtest ──────────────────────────────────────────────────────
    if not args.skip_backtest:
        _run(
            f'5. Backtest  ({cont_bars}-bar continuation)',
            [py, str(SCRIPTS_DIR / 'backtest_btc.py'),
             '--stride',            str(scan_stride),
             '--threshold',         str(args.threshold),
             '--weights',           weights_path,
             '--continuation-bars', str(cont_bars),
             '--output-dir',        backtest_dir],
            env,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Pipeline complete  —  window_size = {N} bars")
    print(f"  All outputs in: {run_dir}")
    print(f"{'='*64}\n")


if __name__ == '__main__':
    main()
