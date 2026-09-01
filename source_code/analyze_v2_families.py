"""
analyze_v2_families.py

Per-family breakdown of a trained v2 model's held-out performance.
The standard evaluation lumps all negatives together; this script attributes
every false positive to the negative family that generated it (walk, channel,
megaphone, stale_wedge) and sweeps decision thresholds per family, answering:

  1. Which hard-negative family fools the model most?
  2. Where should the live threshold sit for a target false-positive rate?

Reads the numpy eval cache built by evaluate_cnn.py (runs it if absent would
be slow — run evaluate_cnn.py first). Writes family_analysis.txt to the run's
models/evaluation/ directory and prints it.

Usage
-----
  python analyze_v2_families.py --data-dir ../runs_v2/window_250bar
  WEDGE_TOTAL_BARS=50 python analyze_v2_families.py \
      --data-dir ../runs_v2/window_50bar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cnn_model import build_model
from train_cnn import TRAIN_FRAC, VAL_FRAC

# eval set = whatever training did not use. Imported rather than hardcoded so
# it can never disagree with evaluate_cnn.py's slice: the assert below checks
# this against a cache built there, and a drifting constant would surface a
# real contamination bug as a confusing shape mismatch instead.
EVAL_FRAC  = TRAIN_FRAC + VAL_FRAC
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main() -> None:
    parser = argparse.ArgumentParser(description='Per-family v2 eval breakdown')
    parser.add_argument('--data-dir', required=True, help='Run directory')
    parser.add_argument('--weights',  default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    root       = Path(args.data_dir)
    cache_dir  = root / 'numpy_cache'
    models_dir = root / 'models'
    out_path   = models_dir / 'evaluation' / 'family_analysis.txt'
    weights    = args.weights or str(models_dir / 'cnn_best.weights.h5')

    # ── Eval arrays (cache built by evaluate_cnn.py) ──────────────────────────
    X = np.load(cache_dir / 'X_eval.npy', mmap_mode='r')
    y = np.load(cache_dir / 'y_eval.npy')

    # ── Family labels for the eval slice, from the manifest ───────────────────
    print('Loading manifest ...')
    with open(root / 'corpus_manifest.json') as fh:
        manifest = json.load(fh)
    manifest.sort(key=lambda e: e['shuffled_idx'])
    eval_start = int(len(manifest) * EVAL_FRAC)
    families   = np.array([e['type'] for e in manifest[eval_start:]])
    labels     = np.array([e['label'] for e in manifest[eval_start:]], dtype=np.float32)

    assert len(families) == X.shape[0], \
        f'manifest eval slice {len(families):,} != cache {X.shape[0]:,}'
    assert np.array_equal(labels, y), 'manifest labels disagree with y_eval cache'

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f'Scoring {X.shape[0]:,} eval windows ...')
    model = build_model(print_summary=False)
    model.load_weights(weights)
    probs = model.predict(X, batch_size=args.batch_size, verbose=1).squeeze()
    np.save(cache_dir / 'eval_probs.npy', probs)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = [f'PER-FAMILY ANALYSIS  ({root.name},  weights={Path(weights).name})',
             '=' * 78]

    # Families come from the manifest, not a hardcoded list: peak-family
    # corpora contain triple_top/double_top/hs_stale and no megaphone or
    # stale_wedge, and a hardcoded name that is absent yields an empty slice
    # that crashes np.percentile. Positive class first, then negatives by
    # descending mean score so the worst confusion is at the top.
    present = sorted(set(families))
    pos_fams = [f for f in present if bool(labels[families == f][0])]
    neg_fams = sorted((f for f in present if f not in pos_fams),
                      key=lambda f: -probs[families == f].mean())
    fam_order = pos_fams + neg_fams

    lines.append(f'\nMean score by family:')
    for fam in fam_order:
        m = families == fam
        lines.append(f'  {fam:<15} n={m.sum():>7,}  mean={probs[m].mean():.4f}  '
                     f'median={np.median(probs[m]):.4f}  p95={np.percentile(probs[m], 95):.4f}')

    lines.append(f'\nFalse-positive rate by family at each threshold '
                 f'(positives: recall):')
    header = f'  {"family":<15}' + ''.join(f'  @{t:<5}' for t in THRESHOLDS)
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))
    for fam in fam_order:
        m    = families == fam
        vals = []
        for t in THRESHOLDS:
            rate = float((probs[m] >= t).mean())
            vals.append(f'  {rate:6.3f}')
        # Which family is the POSITIVE class comes from the manifest labels,
        # not a hardcoded name — the same corpus machinery now builds channel
        # corpora, where forming_wedge is a negative and channel is positive.
        tag = 'recall' if bool(labels[m][0]) else 'FP-rate'
        lines.append(f'  {fam:<15}' + ''.join(vals) + f'   ({tag})')

    lines.append(f'\nOverall precision at each threshold:')
    pos = labels == 1
    for t in THRESHOLDS:
        called = probs >= t
        prec   = float(labels[called].mean()) if called.any() else float('nan')
        lines.append(f'  @{t:<5}  precision={prec:.4f}  '
                     f'recall={float(called[pos].mean()):.4f}  '
                     f'n_called={int(called.sum()):,}')

    report = '\n'.join(lines)
    print('\n' + report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
