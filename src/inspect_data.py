from pathlib import Path
import pandas as pd

# Location of our raw data
DATA_DIR = Path("data/raw")

# Find all CSV files
files = list(DATA_DIR.glob("*.csv"))

print("=" * 60)
print("FOOTBALL DATASET INSPECTION")
print("=" * 60)

print(f"\nCSV files found: {len(files)}")

for file in files:
    print(f"  - {file.name}")

# Load all files
dataframes = []

for file in files:
    print(f"\nLoading: {file.name}")

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    dataframes.append(df)

# Combine everything
data = pd.concat(dataframes, ignore_index=True)

print("\n" + "=" * 60)
print("COMBINED DATASET")
print("=" * 60)

print(f"\nTotal matches: {len(data):,}")
print(f"Total columns: {len(data.columns)}")

print("\nDate range:")

dates = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")

print(f"Earliest: {dates.min()}")
print(f"Latest:   {dates.max()}")

print("\nUnique teams:")
print(data["HomeTeam"].nunique())

print("\nTeams:")
teams = sorted(
    set(data["HomeTeam"].dropna())
    | set(data["AwayTeam"].dropna())
)

for team in teams:
    print(f"  {team}")

print("\nResult distribution:")
print(data["FTR"].value_counts())

print("\nResult percentages:")
print(data["FTR"].value_counts(normalize=True).round(4) * 100)

print("\nMissing values in important columns:")

important_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
    "AvgH",
    "AvgD",
    "AvgA",
]

for column in important_columns:
    if column in data.columns:
        missing = data[column].isna().sum()
        percentage = missing / len(data) * 100

        print(
            f"{column:12} "
            f"{missing:6,} missing "
            f"({percentage:.2f}%)"
        )

print("\nDuplicate rows:")
print(data.duplicated().sum())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)