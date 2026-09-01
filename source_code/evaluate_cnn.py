"""
evaluate_cnn.py

Reconstructs the 1D CNN from saved weights and evaluates it against the
held-out 100,000 datasets (shuffled_idx 400,000 - 499,999 in the corpus).

Outputs (written to models/evaluation/)
-----------------------------------------
  classification_report.txt  -- precision / recall / F1 per class + ROC/PR AUC
  confusion_matrix.png
  roc_pr_curves.png          -- ROC curve and Precision-Recall curve side-by-side
  score_distribution.png     -- predicted probability histograms by true class

Prerequisites
-------------
  Run train_cnn.py first to produce models/cnn_best.weights.h5.

Usage
-----
  python evaluate_cnn.py
  python evaluate_cnn.py --weights ../models/cnn_final.weights.h5
  python evaluate_cnn.py --threshold 0.4   # adjust decision boundary
  python evaluate_cnn.py --no-cache        # force reload from parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import tensorflow as tf
from tensorflow import keras

from cnn_model import build_model, FEATURE_COLS, N_BARS, N_FEATURES
# Split fractions come from train_cnn so the eval boundary can never drift
# from the boundary training actually used.
from train_cnn import TRAIN_FRAC, VAL_FRAC


# =============================================================================
# Data loading  (mirrors train_cnn.py helpers)
# =============================================================================

def _file_path(entry: dict, root: Path) -> Path:
    subdir = 'training_data' if entry['split'] == 'training' else 'validation_data'
    return root / subdir / entry['filename']


def load_from_parquet(
    entries: list[dict],
    root: Path,
    label: str = 'eval',
) -> tuple[np.ndarray, np.ndarray]:
    n    = len(entries)
    X    = np.empty((n, N_BARS, N_FEATURES), dtype=np.float32)
    y    = np.empty(n, dtype=np.float32)
    cols = FEATURE_COLS + ['label']

    t0 = time.time()
    for i, entry in enumerate(entries):
        df   = pd.read_parquet(_file_path(entry, root), columns=cols)
        X[i] = df[FEATURE_COLS].values
        y[i] = df['label'].iat[0]
        if (i + 1) % 10_000 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            print(f'  [{label}] {i+1:>7,}/{n:,}  '
                  f'({rate:,.0f} files/s  ~{remaining/60:.1f} min left)')
    return X, y


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    return cache_dir / f'X_{split}.npy', cache_dir / f'y_{split}.npy'


def load_cache(
    cache_dir: Path, split: str, expected_n: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    X_path, y_path = _cache_paths(cache_dir, split)
    if not (X_path.exists() and y_path.exists()):
        return None, None
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path)
    if X.shape[0] != expected_n:
        print(f'  Cache size mismatch ({X.shape[0]:,} vs {expected_n:,}) — reloading.')
        return None, None
    print(f'  Loaded {split} from cache: X={X.shape}  y={y.shape}')
    return X, y


def save_cache(X: np.ndarray, y: np.ndarray, cache_dir: Path, split: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_path, y_path = _cache_paths(cache_dir, split)
    np.save(X_path, X)
    np.save(y_path, y)
    print(f'  Saved {split} cache: {X_path}')


# =============================================================================
# Evaluation plots
# =============================================================================

_FIG_BG = '#1a1a2e'
_AX_BG  = '#13131f'
_GRID   = '#2a2a3e'
_SPINE  = '#333345'
_MUTED  = '#8a8aa0'
_TEXT   = '#d0d0e8'
_BLUE   = '#42A5F5'
_GREEN  = '#26a69a'
_RED    = '#ef5350'


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.20, color=_GRID)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AX_BG)

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes    = ['Noise (0)', 'Wedge (1)']
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, color=_TEXT, fontsize=11)
    ax.set_yticklabels(classes, color=_TEXT, fontsize=11)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            val     = cm[i, j]
            txt_col = 'white' if val > cm.max() / 2 else _TEXT
            ax.text(j, i, f'{val:,}\n({val / total * 100:.1f}%)',
                    ha='center', va='center', color=txt_col, fontsize=11)

    ax.set_ylabel('True Label',      color=_TEXT, fontsize=11)
    ax.set_xlabel('Predicted Label', color=_TEXT, fontsize=11)
    ax.set_title('Confusion Matrix', color=_TEXT, fontsize=13)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Confusion matrix saved: {out_path}')


def plot_roc_pr(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path
) -> None:
    fpr, tpr, _   = roc_curve(y_true, y_prob)
    roc_auc_score = auc(fpr, tpr)

    prec, rec, _  = precision_recall_curve(y_true, y_prob)
    pr_auc_score  = auc(rec, prec)
    baseline_prec = float(y_true.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_FIG_BG)
    _style(ax1)
    _style(ax2)

    # ROC
    ax1.plot(fpr, tpr, color=_BLUE, lw=2,
             label=f'ROC AUC = {roc_auc_score:.4f}')
    ax1.plot([0, 1], [0, 1], color=_MUTED, lw=1, ls='--', label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.10, color=_BLUE)
    ax1.set_xlabel('False Positive Rate', color=_MUTED, fontsize=10)
    ax1.set_ylabel('True Positive Rate',  color=_MUTED, fontsize=10)
    ax1.set_title('ROC Curve', color=_TEXT, fontsize=12)
    ax1.legend(facecolor='#222233', labelcolor=_TEXT, fontsize=9)

    # Precision-Recall
    ax2.plot(rec, prec, color=_GREEN, lw=2,
             label=f'PR AUC = {pr_auc_score:.4f}')
    ax2.axhline(baseline_prec, color=_MUTED, lw=1, ls='--',
                label=f'Random baseline = {baseline_prec:.3f}')
    ax2.fill_between(rec, prec, alpha=0.10, color=_GREEN)
    ax2.set_xlabel('Recall',    color=_MUTED, fontsize=10)
    ax2.set_ylabel('Precision', color=_MUTED, fontsize=10)
    ax2.set_title('Precision-Recall Curve', color=_TEXT, fontsize=12)
    ax2.legend(facecolor='#222233', labelcolor=_TEXT, fontsize=9)

    fig.suptitle('Model Evaluation — 1D CNN Rising Wedge Classifier',
                 color='#e8e8ff', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'ROC + PR curves saved: {out_path}')


def plot_score_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_FIG_BG)
    _style(ax)

    bins = np.linspace(0, 1, 51)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.70, color=_BLUE,
            label='Noise (label=0)', density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.70, color=_GREEN,
            label='Wedge (label=1)', density=True)
    ax.axvline(threshold, color=_RED, lw=1.8, ls='--',
               label=f'Decision threshold ({threshold})')

    ax.set_xlabel('Predicted Probability (wedge)', color=_MUTED, fontsize=10)
    ax.set_ylabel('Density', color=_MUTED, fontsize=10)
    ax.set_title('Score Distribution by True Class', color=_TEXT, fontsize=12)
    ax.legend(facecolor='#222233', labelcolor=_TEXT, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Score distribution saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate 1D CNN Rising Wedge Classifier on held-out data'
    )
    parser.add_argument('--data-dir',   default='..', help='Project root directory')
    parser.add_argument('--weights',    default=None,
                        help='Weights file (default: ../models/cnn_best.weights.h5)')
    parser.add_argument('--eval-start', type=int, default=None,
                        help='First manifest shuffled_idx in the eval set '
                             '(default: derived from the corpus size -- the '
                             'same train+val boundary train_cnn.py used)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Decision threshold for binary prediction (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--no-cache',   action='store_true',
                        help='Force reload from parquet')
    args = parser.parse_args()

    root         = Path(args.data_dir)
    models_dir   = root / 'models'
    cache_dir    = root / 'numpy_cache'
    eval_out_dir = models_dir / 'evaluation'
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.weights or str(models_dir / 'cnn_best.weights.h5')
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f'Weights file not found: {weights_path}\n'
            f'Run train_cnn.py first.'
        )

    # ── Load manifest and slice eval entries ──────────────────────────────────
    manifest_path = root / 'corpus_manifest.json'
    print('Loading manifest ...')
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    manifest.sort(key=lambda e: e['shuffled_idx'])

    # The train/val/eval split is FRACTIONAL (train_cnn.py slices the shuffled
    # manifest at TRAIN_FRAC and TRAIN_FRAC+VAL_FRAC), so the eval boundary
    # moves with corpus size. The old default of 400,000 was an absolute index
    # from the 500K v1 corpus and silently leaks training data into the eval
    # set on every other corpus: on the 700K v3 corpus it made 53% of the eval
    # set contaminated, and on a 1M corpus it would be 67%. Derive it instead.
    boundary   = int(len(manifest) * (TRAIN_FRAC + VAL_FRAC))
    eval_start = args.eval_start
    if eval_start is None:
        eval_start = boundary
        print(f'  eval-start derived from corpus size: {eval_start:,} '
              f'(of {len(manifest):,} entries)')
    elif eval_start < boundary:
        leaked = boundary - eval_start
        print(f'  WARNING: --eval-start {eval_start:,} is below the train+val '
              f'boundary {boundary:,} -- {leaked:,} training/validation '
              f'windows ({leaked/(len(manifest)-eval_start)*100:.0f}% of this '
              f'eval set) are CONTAMINATED.')

    eval_entries = manifest[eval_start:]
    n_eval       = len(eval_entries)
    wedge_count  = sum(1 for e in eval_entries if e['label'] == 1)
    noise_count  = n_eval - wedge_count

    print(f'Eval set: {n_eval:,} datasets '
          f'(wedge={wedge_count:,}, noise={noise_count:,})')
    print(f'Using weights: {weights_path}\n')

    # ── Load eval data ────────────────────────────────────────────────────────
    X_eval, y_eval = (None, None) if args.no_cache else load_cache(
        cache_dir, 'eval', n_eval
    )
    if X_eval is None:
        print('Loading eval data from parquet files ...')
        X_eval, y_eval = load_from_parquet(eval_entries, root, label='eval')
        save_cache(X_eval, y_eval, cache_dir, 'eval')

    print(f'\nX_eval: {X_eval.shape}  y_eval: {y_eval.shape}')

    # ── Reconstruct model and load weights ────────────────────────────────────
    print(f'\nReconstructing model and loading weights ...')
    model = build_model(print_summary=False)
    model.load_weights(weights_path)
    print('Weights loaded successfully.\n')

    # ── Predict ───────────────────────────────────────────────────────────────
    print(f'Running inference on {n_eval:,} samples ...')
    y_prob = model.predict(
        X_eval, batch_size=args.batch_size, verbose=1
    ).squeeze()
    y_pred = (y_prob >= args.threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    fpr, tpr, _   = roc_curve(y_eval, y_prob)
    roc_auc_score = auc(fpr, tpr)
    prec, rec, _  = precision_recall_curve(y_eval, y_prob)
    pr_auc_score  = auc(rec, prec)

    report = classification_report(
        y_eval, y_pred,
        target_names=['Noise (0)', 'Wedge (1)'],
        digits=4,
    )
    cm = confusion_matrix(y_eval, y_pred)
    tn, fp, fn, tp = cm.ravel()

    summary = (
        f'{"="*56}\n'
        f'EVALUATION RESULTS\n'
        f'{"="*56}\n'
        f'Weights    : {weights_path}\n'
        f'Threshold  : {args.threshold}\n'
        f'Eval size  : {n_eval:,}  '
        f'(wedge={wedge_count:,}, noise={noise_count:,})\n'
        f'\n'
        f'ROC AUC    : {roc_auc_score:.4f}\n'
        f'PR  AUC    : {pr_auc_score:.4f}\n'
        f'\n'
        f'{report}\n'
        f'Confusion matrix (threshold={args.threshold}):\n'
        f'  TN = {tn:>7,}   FP = {fp:>7,}\n'
        f'  FN = {fn:>7,}   TP = {tp:>7,}\n'
        f'{"="*56}\n'
    )

    print(summary)

    report_path = eval_out_dir / 'classification_report.txt'
    report_path.write_text(summary)
    print(f'Report saved: {report_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, eval_out_dir / 'confusion_matrix.png')
    plot_roc_pr(y_eval, y_prob,   eval_out_dir / 'roc_pr_curves.png')
    plot_score_distribution(
        y_eval, y_prob, args.threshold,
        eval_out_dir / 'score_distribution.png',
    )

    print(f'\nAll evaluation outputs written to: {eval_out_dir}')


if __name__ == '__main__':
    main()
