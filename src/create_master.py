from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "epl_master.csv"

# Map each file to its Premier League season
SEASON_MAP = {
    "E0 (9).csv": "2016/17",
    "E0 (8).csv": "2017/18",
    "E0 (7).csv": "2018/19",
    "E0 (6).csv": "2019/20",
    "E0 (5).csv": "2020/21",
    "E0 (4).csv": "2021/22",
    "E0 (3).csv": "2022/23",
    "E0 (2).csv": "2023/24",
    "E0 (1).csv": "2024/25",
    "E0.csv": "2025/26",
}

# ============================================================
# LOAD FILES
# ============================================================

files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("CREATING EPL MASTER DATASET")
print("=" * 80)

print(f"\nCSV files found: {len(files)}")

all_data = []

for file in files:

    print(f"\nLoading: {file.name}")

    df = pd.read_csv(file)

    # Add source file
    df["source_file"] = file.name

    # Add season
    df["season"] = SEASON_MAP.get(file.name, "UNKNOWN")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Season: {df['season'].iloc[0]}")

    all_data.append(df)

# ============================================================
# COMBINE
# ============================================================

master = pd.concat(
    all_data,
    ignore_index=True,
    sort=False
)

print("\n" + "=" * 80)
print("COMBINED DATASET")
print("=" * 80)

print(f"Rows:    {len(master):,}")
print(f"Columns: {len(master.columns):,}")

# ============================================================
# PARSE DATES
# ============================================================

print("\nParsing dates...")

def parse_date(value):
    """
    Handles both date formats found in the dataset:

    Newer seasons:
        16/08/2024

    2016/17 season:
        13/08/16
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    # Determine whether the year has 2 or 4 digits
    year_part = value.split("/")[-1]

    if len(year_part) == 4:
        return pd.to_datetime(
            value,
            format="%d/%m/%Y",
            errors="coerce"
        )

    elif len(year_part) == 2:
        return pd.to_datetime(
            value,
            format="%d/%m/%y",
            errors="coerce"
        )

    return pd.NaT


master["Date"] = master["Date"].apply(parse_date)

missing_dates = master["Date"].isna().sum()

print(f"Missing dates: {missing_dates}")

# ============================================================
# CREATE MATCH ID
# ============================================================

master["match_id"] = (
    master["Date"].dt.strftime("%Y-%m-%d")
    + "_"
    + master["HomeTeam"].astype(str)
    + "_"
    + master["AwayTeam"].astype(str)
)

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

master = master.sort_values(
    ["Date", "HomeTeam", "AwayTeam"]
).reset_index(drop=True)

# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_count = master["match_id"].duplicated().sum()

unique_matches = master["match_id"].nunique()

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

print(f"Total rows:       {len(master):,}")
print(f"Unique match IDs: {unique_matches:,}")
print(f"Duplicate rows:   {duplicate_count:,}")

# ============================================================
# SEASON CHECK
# ============================================================

print("\n" + "=" * 80)
print("SEASON SUMMARY")
print("=" * 80)

season_summary = (
    master.groupby("season")
    .agg(
        matches=("match_id", "count"),
        unique_matches=("match_id", "nunique"),
        first_date=("Date", "min"),
        last_date=("Date", "max")
    )
    .reset_index()
)

print(season_summary.to_string(index=False))

# ============================================================
# TEAM CHECK
# ============================================================

teams = set(master["HomeTeam"].dropna()) | set(
    master["AwayTeam"].dropna()
)

print("\n" + "=" * 80)
print("TEAM SUMMARY")
print("=" * 80)

print(f"Unique teams across all seasons: {len(teams)}")

# ============================================================
# RESULT CHECK
# ============================================================

print("\n" + "=" * 80)
print("RESULT DISTRIBUTION")
print("=" * 80)

print(master["FTR"].value_counts())

# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

master.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("MASTER DATASET CREATED")
print("=" * 80)

print(f"File: {OUTPUT_FILE}")
print(f"Rows: {len(master):,}")
print(f"Columns: {len(master.columns):,}")