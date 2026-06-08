# Claude Technical Analysis

A project for technical analysis using Claude AI.

## Structure

- `reference_material/` — documentation, papers, and reference resources
- `source_code/` — application and analysis source code
- `training_data/` — datasets used for training
- `validation_data/` — datasets used for validation and evaluation

## Daily chart

`source_code/plot_daily.py` renders `rising_wedge_chart.png`, overlaying the
day's SPY price (`spy_data_1min.csv`) with the live model output
(`rising_wedge.csv`): a price panel with markers where 50- and 250-bar signals
fired, and a score panel with the decision threshold. The deployment's 7 PM ET
cron job (`deploy/push_results.sh`) regenerates it before pushing, so the latest
chart is committed alongside the CSVs.

```bash
python plot_daily.py                  # most recent trading day (default)
python plot_daily.py --date 2026-06-08   # a specific day
python plot_daily.py --all            # entire history
python plot_daily.py --threshold 0.65 # custom signal threshold
```

Run from `source_code/`, or pass `--data-dir <project root>` from elsewhere.
If no price data exists yet, the script prints a notice and exits without
writing a chart.
