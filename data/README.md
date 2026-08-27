# Data

This project uses the **[Smart Meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london)**
dataset (Kaggle, published by UK Power Networks / London Datastore under the UK Open
Government Licence): half-hourly electricity readings for 5,567 London households
between November 2011 and February 2014, from the UK Power Networks "Low Carbon London"
project.

## What's included in this repo

The full dataset is **~12 GB**, which is too large to host in a git repository. This
folder instead ships a **reproducible sample** so the notebook/script run out of the
box:

| File | Contents | Size |
|---|---|---|
| `daily_dataset.csv` | Daily energy stats (median/mean/max/min/std/sum/count) for a random sample of **400 of the 5,566 households** (seed 42), full date range | ~23 MB |
| `informations_households.csv` | Household ID, tariff (`std`/`ToU`), Acorn/Acorn_grouped demographic group — for the 400 sampled households | ~17 KB |
| `acorn_details.csv` | Full CACI Acorn category reference table | ~130 KB |
| `uk_bank_holidays.csv` | Full list of UK bank holidays in the study period | ~1 KB |
| `weather_daily_darksky.csv` | Full daily weather (London) for the study period | ~340 KB |
| `weather_hourly_darksky.csv` | Full hourly weather (London) for the study period | ~2 MB |

Running `electricity_clustering_analysis.py --data-dir data` against these files
reproduces the full pipeline (cleaning → feature engineering → clustering → evaluation
→ recommendations) on a smaller, GitHub-friendly scale. Cluster sizes/positions will
differ slightly from the headline results in the main README, which were computed on
the **complete** dataset — see below.

## Getting the full dataset

The figures in [`/figures`](../figures) and the reference output in
[`/outputs`](../outputs) were generated from the complete dataset. To reproduce them
exactly:

1. Download **Smart Meters in London** from Kaggle:
   https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london
2. Unzip it. You'll get `daily_dataset.csv`, `informations_households.csv`,
   `acorn_details.csv`, `uk_bank_holidays.csv`, `weather_daily_darksky.csv`,
   `weather_hourly_darksky.csv`, plus two large sub-folders:
   - `halfhourly_dataset/` (~7.4 GB) — raw half-hourly readings, not used by this project
   - `hhblock_dataset/` (~1.6 GB) — the same half-hourly readings reshaped into 48
     `hh_0`...`hh_47` columns per household per day, used **only** for the optional
     true hour-of-day peak/off-peak analysis (Step 11 / `--hhblock-dir`)
3. Run:
   ```bash
   python electricity_clustering_analysis.py \
       --data-dir /path/to/full/dataset \
       --hhblock-dir /path/to/full/dataset/hhblock_dataset \
       --best-k 4
   ```

## Column reference

**`daily_dataset.csv`**

| Column | Meaning |
|---|---|
| `LCLid` | Household ID |
| `day` | Calendar date |
| `energy_median`, `energy_mean`, `energy_max`, `energy_min`, `energy_std` | Summary stats of that household's half-hourly readings for the day (kWh) |
| `energy_sum` | Total energy consumed that day (kWh) |
| `energy_count` | Number of half-hourly readings recorded that day (out of 48) |

**`informations_households.csv`**

| Column | Meaning |
|---|---|
| `LCLid` | Household ID |
| `stdorToU` | Tariff type: flat-rate (`Std`) or dynamic Time-of-Use (`ToU`) |
| `Acorn` | Detailed CACI Acorn demographic category |
| `Acorn_grouped` | Coarser Acorn group: Affluent / Comfortable / Adversity / ACORN-U (unclassified) |
| `file` | Source block file in the original dataset |

**`hhblock_dataset/block_*.csv`** (not bundled — see above)

`LCLid`, `day`, `hh_0` ... `hh_47` — one column per half-hour of the day (`hh_0` =
00:00–00:30, `hh_47` = 23:30–24:00), values in kWh, `"Null"` for missing readings.

## License / attribution

Data © UK Power Networks, distributed via the London Datastore and mirrored on Kaggle
under the UK Open Government Licence. Not redistributed here beyond the small sample
above for reproducibility; see the Kaggle page for full terms.
