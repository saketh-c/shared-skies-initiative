"""Post-process the output of 06_pull_purpleair_full.py to perfectly match the
column names that pipeline/03_train_enhanced.py expects, so the new dataset can be
swapped in for p2_processed.xls without touching the training code.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pipeline" / "purpleair_full_dataset.parquet"
OUT_PARQUET = ROOT / "pipeline" / "purpleair_training_ready.parquet"
OUT_CSV = ROOT / "pipeline" / "purpleair_training_ready.csv"
# Drop-in replacement for p2_processed.xls (CSV format, original .xls extension)
OUT_LEGACY = ROOT / "p2_processed_v2.xls"

EXPECTED = [
    "pm25", "humidity", "temperature", "pressure", "wind_speed", "precipitation",
    "ejf_score", "pct_people_of_color", "pct_low_income",
    "traffic_proximity", "superfund_proximity", "rmp_proximity",
    "diesel_pm_proximity", "pct_ling_isolated",
    "latitude", "longitude",
    "month", "hour", "dow", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "temp_x_humidity", "wind_x_temp",
]


def main():
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {SRC}")

    if "doy" in df.columns and "day_of_year" not in df.columns:
        df = df.rename(columns={"doy": "day_of_year"})

    # Sanity: which expected features are present
    missing = [c for c in EXPECTED if c not in df.columns]
    present = [c for c in EXPECTED if c in df.columns]
    print(f"Expected features present: {len(present)}/{len(EXPECTED)}")
    if missing:
        print(f"Still missing: {missing}")

    # Backwards-compat columns the legacy training script reads
    if "season" not in df.columns:
        season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
        df["season"] = df["month"].map(season_map)
    if "city" not in df.columns:
        df["city"] = ""

    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    df.to_csv(OUT_LEGACY, index=False)
    print(f"\nWrote:")
    print(f"  {OUT_PARQUET}  ({len(df):,} rows)")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_LEGACY}  (drop-in replacement for p2_processed.xls)")
    print(f"Sensors: {df['sensor_id'].nunique()}")
    print(f"Date range: {pd.to_datetime(df['date']).min().date()} → {pd.to_datetime(df['date']).max().date()}")
    if "pm25" in df.columns:
        print(f"PM2.5: min {df['pm25'].min():.2f}, mean {df['pm25'].mean():.2f}, "
              f"max {df['pm25'].max():.2f}, p99 {df['pm25'].quantile(0.99):.2f}")


if __name__ == "__main__":
    main()
