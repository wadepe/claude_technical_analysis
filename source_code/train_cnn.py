"""
train_cnn.py

Loads the first 400,000 datasets from the mixed corpus, trains the 1D CNN
binary classifier, and saves the best weights for use by evaluate_cnn.py.

Data flow
---------
  corpus_manifest.json  (shuffled_idx 0 - 499,999)
        |
        +-- entries 0 - 359,999  -> Keras training set   (360K)
        +-- entries 360,000 - 399,999  -> Keras validation set  (40K)
        +-- entries 400,000 - 499,999  -> held-out eval (used by evaluate_cnn.py only)

First run
---------
  All parquet files are read and saved as numpy arrays in numpy_cache/.
  This one-time step takes ~15-25 minutes depending on disk speed.
  Subsequent runs load from the cache in seconds.

Prerequisites
-------------
  1. Generate the mixed corpus:
       python generate_rising_wedge.py --corpus
  2. Install dependencies:
       pip install tensorflow scikit-learn pandas pyarrow matplotlib

Usage
-----
  python train_cnn.py
  python train_cnn.py --epochs 50 --batch-size 256
  python train_cnn.py --no-cache       # force reload from parquet
  python train_cnn.py --preprocess-only  # build numpy cache then exit
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
import tensorflow as tf
from tensorflow import keras

from cnn_model import build_model, FEATURE_COLS, N_BARS, N_FEATURES


# =============================================================================
# Constants
# =============================================================================

TRAIN_END  = 360_000   # entries 0       - 359,999  used for Keras .fit()
VAL_END    = 400_000   # entries 360,000 - 399,999  used for Keras validation
# entries 400,000 - 499,999  are the held-out eval set (evaluate_cnn.py)


# =============================================================================
# Data loading helpers
# =============================================================================

def _file_path(entry: dict, root: Path) -> Path:
    subdir = 'training_data' if entry['split'] == 'training' else 'validation_data'
    return root / subdir / entry['filename']


def load_from_parquet(
    entries: list[dict],
    root: Path,
    label: str = 'data',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read FEATURE_COLS + label from each parquet file.

    Returns
    -------
    X : float32 array, shape (n, N_BARS, N_FEATURES)
    y : float32 array, shape (n,)
    """
    n = len(entries)
    X = np.empty((n, N_BARS, N_FEATURES), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    cols = FEATURE_COLS + ['label']

    t0 = time.time()
    for i, entry in enumerate(entries):
        df    = pd.read_parquet(_file_path(entry, root), columns=cols)
        X[i]  = df[FEATURE_COLS].values
        y[i]  = df['label'].iat[0]

        if (i + 1) % 10_000 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            print(f'  [{label}] {i+1:>7,}/{n:,} '
                  f'({rate:,.0f} files/s  ~{remaining/60:.1f} min left)')

    return X, y


def _cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path]:
    return cache_dir / f'X_{split}.npy', cache_dir / f'y_{split}.npy'


def load_cache(cache_dir: Path, split: str, expected_n: int
               ) -> tuple[np.ndarray | None, np.ndarray | None]:
    X_path, y_path = _cache_paths(cache_dir, split)
    if not (X_path.exists() and y_path.exists()):
        return None, None
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path)
    if X.shape[0] != expected_n:
        print(f'  Cache shape mismatch for {split} '
              f'(cached {X.shape[0]:,}, expected {expected_n:,}) — reloading.')
        return None, None
    print(f'  Loaded {split} from cache: X={X.shape}  y={y.shape}')
    return X, y


def save_cache(X: np.ndarray, y: np.ndarray, cache_dir: Path, split: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_path, y_path = _cache_paths(cache_dir, split)
    np.save(X_path, X)
    np.save(y_path, y)
    print(f'  Saved {split} cache: {X_path}')


def get_split(
    entries: list[dict],
    root: Path,
    cache_dir: Path,
    split_name: str,
    force_reload: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split from cache if available, otherwise read parquet files."""
    if not force_reload:
        X, y = load_cache(cache_dir, split_name, len(entries))
        if X is not None:
            return X, y
    print(f'Loading {split_name} set ({len(entries):,} files) from parquet ...')
    X, y = load_from_parquet(entries, root, label=split_name)
    save_cache(X, y, cache_dir, split_name)
    return X, y


# =============================================================================
# Training helpers
# =============================================================================

def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights to compensate for 4:1 noise:wedge imbalance."""
    n_total = len(y)
    n_pos   = int(y.sum())
    n_neg   = n_total - n_pos
    return {
        0: n_total / (2.0 * n_neg),
        1: n_total / (2.0 * n_pos),
    }


def save_history_plot(history: keras.callbacks.History, out_path: Path) -> None:
    metrics = [
        ('loss',      'Binary Cross-Entropy Loss'),
        ('accuracy',  'Accuracy'),
        ('auc',       'ROC AUC'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor('#13131f')
        ax.tick_params(colors='#8a8aa0', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333345')
        ax.grid(True, alpha=0.20, color='#2a2a3e')

        ax.plot(history.history[key],           color='#42A5F5', lw=1.8, label='Train')
        ax.plot(history.history[f'val_{key}'],  color='#26a69a', lw=1.8, label='Val')
        ax.set_title(title, color='#d0d0e8', fontsize=10)
        ax.set_xlabel('Epoch', color='#8a8aa0', fontsize=8)
        ax.legend(facecolor='#222233', labelcolor='#d0d0e8', fontsize=8)

    fig.suptitle('Training History — 1D CNN Rising Wedge Classifier',
                 color='#e8e8ff', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'History plot saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train 1D CNN Rising Wedge Classifier'
    )
    parser.add_argument('--data-dir',        default='..', help='Project root directory')
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--batch-size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--no-cache',        action='store_true',
                        help='Force reload from parquet even if numpy cache exists')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Build numpy cache files then exit without training')
    args = parser.parse_args()

    root       = Path(args.data_dir)
    models_dir = root / 'models'
    cache_dir  = root / 'numpy_cache'
    models_dir.mkdir(exist_ok=True)

    # ── Load and validate manifest ────────────────────────────────────────────
    manifest_path = root / 'corpus_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Corpus manifest not found at {manifest_path}\n'
            f'Generate the corpus first:\n'
            f'  python generate_rising_wedge.py --corpus'
        )

    print('Loading manifest ...')
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    manifest.sort(key=lambda e: e['shuffled_idx'])
    print(f'  Total entries in manifest: {len(manifest):,}')

    if len(manifest) < VAL_END:
        raise ValueError(
            f'Manifest has only {len(manifest):,} entries; '
            f'need at least {VAL_END:,} for training. '
            f'Re-run corpus generation.'
        )

    train_entries = manifest[:TRAIN_END]
    val_entries   = manifest[TRAIN_END:VAL_END]

    n_wedge_train = sum(1 for e in train_entries if e['label'] == 1)
    n_wedge_val   = sum(1 for e in val_entries   if e['label'] == 1)
    print(f'  Keras train : {len(train_entries):,}  '
          f'(wedge={n_wedge_train:,}, noise={len(train_entries)-n_wedge_train:,})')
    print(f'  Keras val   : {len(val_entries):,}  '
          f'(wedge={n_wedge_val:,},   noise={len(val_entries)-n_wedge_val:,})')
    print(f'  Held-out    : {len(manifest) - VAL_END:,}  '
          f'(evaluate_cnn.py)\n')

    # ── Load data (from cache or parquet) ─────────────────────────────────────
    force = args.no_cache
    X_train, y_train = get_split(train_entries, root, cache_dir, 'train', force)
    X_val,   y_val   = get_split(val_entries,   root, cache_dir, 'val',   force)

    if args.preprocess_only:
        print('\nPreprocessing complete. Exiting (--preprocess-only).')
        return

    print(f'\nX_train: {X_train.shape}  y_train: {y_train.shape}')
    print(f'X_val  : {X_val.shape}    y_val  : {y_val.shape}')

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weight = compute_class_weights(y_train)
    print(f'\nClass weights: '
          f'noise={class_weight[0]:.4f}  wedge={class_weight[1]:.4f}')

    # ── Build and compile model ───────────────────────────────────────────────
    print('\nBuilding model ...')
    model = build_model(print_summary=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_weights_path  = str(models_dir / 'cnn_best.weights.h5')
    final_weights_path = str(models_dir / 'cnn_final.weights.h5')

    callbacks = [
        # Save the weights that achieve the best validation AUC
        keras.callbacks.ModelCheckpoint(
            filepath    = best_weights_path,
            monitor     = 'val_auc',
            mode        = 'max',
            save_best_only   = True,
            save_weights_only= True,
            verbose     = 1,
        ),
        # Stop early if val_auc does not improve for 7 consecutive epochs
        keras.callbacks.EarlyStopping(
            monitor             = 'val_auc',
            mode                = 'max',
            patience            = 7,
            restore_best_weights= True,
            verbose             = 1,
        ),
        # Halve the learning rate after 3 stagnant epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_auc',
            mode     = 'max',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        # Per-epoch CSV log
        keras.callbacks.CSVLogger(
            str(models_dir / 'training_log.csv'),
            append=False,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f'\nTraining up to {args.epochs} epochs '
          f'(early stopping patience=7 on val_auc) ...\n')

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        class_weight    = class_weight,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # ── Save final weights ────────────────────────────────────────────────────
    model.save_weights(final_weights_path)
    print(f'\nBest weights  -> {best_weights_path}')
    print(f'Final weights -> {final_weights_path}')

    # ── Training summary ──────────────────────────────────────────────────────
    val_aucs  = history.history['val_auc']
    best_ep   = int(np.argmax(val_aucs)) + 1
    best_auc  = float(np.max(val_aucs))
    best_acc  = float(history.history['val_accuracy'][best_ep - 1])
    best_prec = float(history.history['val_precision'][best_ep - 1])
    best_rec  = float(history.history['val_recall'][best_ep - 1])

    print(f'\n{"="*50}')
    print(f'TRAINING COMPLETE')
    print(f'{"="*50}')
    print(f'Best epoch     : {best_ep}')
    print(f'Val AUC        : {best_auc:.4f}')
    print(f'Val Accuracy   : {best_acc:.4f}')
    print(f'Val Precision  : {best_prec:.4f}')
    print(f'Val Recall     : {best_rec:.4f}')
    print(f'{"="*50}')

    save_history_plot(history, models_dir / 'training_history.png')
    print('\nRun evaluate_cnn.py to assess the held-out 100,000 datasets.')


if __name__ == '__main__':
    main()
