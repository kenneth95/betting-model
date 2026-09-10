from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Load all CSV files
files = list(DATA_DIR.glob("*.csv"))

dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("=" * 70)
print("COLUMN ANALYSIS")
print("=" * 70)

print(f"\nTotal matches: {len(data):,}")
print(f"Total columns: {len(data.columns)}")

# ---------------------------------------------------------
# 1. PRINT ALL COLUMNS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ALL COLUMNS")
print("=" * 70)

for i, column in enumerate(data.columns, start=1):
    print(f"{i:3}. {column}")

# ---------------------------------------------------------
# 2. COLUMN TYPES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN CATEGORIES")
print("=" * 70)

categories = {
    "Match Information": [
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "Referee",
    ],

    "Final Result": [
        "FTHG",
        "FTAG",
        "FTR",
    ],

    "Half-Time Result": [
        "HTHG",
        "HTAG",
        "HTR",
    ],
}

for category, columns in categories.items():

    existing = [c for c in columns if c in data.columns]

    print(f"\n{category}:")

    for column in existing:
        print(f"  {column}")

# ---------------------------------------------------------
# 3. BOOKMAKER ODDS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BOOKMAKER / MARKET ODDS")
print("=" * 70)

odds_columns = []

for column in data.columns:

    if any(x in column for x in [
        "B365",
        "BW",
        "BF",
        "PS",
        "WH",
        "1XB",
        "Max",
        "Avg",
        "BFE",
    ]):
        odds_columns.append(column)

for column in odds_columns:
    print(f"  {column}")

print(f"\nTotal odds-related columns: {len(odds_columns)}")

# ---------------------------------------------------------
# 4. MATCH STATISTICS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MATCH STATISTICS")
print("=" * 70)

stats_columns = [
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

for column in stats_columns:

    if column in data.columns:

        missing = data[column].isna().sum()

        print(
            f"  {column:6} "
            f"missing: {missing:,} "
            f"({missing / len(data) * 100:.1f}%)"
        )

# ---------------------------------------------------------
# 5. MISSING DATA
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING DATA")
print("=" * 70)

missing = data.isna().sum()

missing = missing[missing > 0].sort_values(ascending=False)

for column, count in missing.items():

    percentage = count / len(data) * 100

    print(
        f"{column:15} "
        f"{count:6,} missing "
        f"({percentage:5.1f}%)"
    )

# ---------------------------------------------------------
# 6. CLOSING ODDS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLOSING ODDS")
print("=" * 70)

closing_columns = [
    column
    for column in data.columns
    if column.startswith("B365C")
    or column.startswith("BWC")
    or column.startswith("BFC")
    or column.startswith("PSC")
    or column.startswith("WHC")
    or column.startswith("1XBC")
    or column.startswith("MaxC")
    or column.startswith("AvgC")
    or column.startswith("BFEC")
]

for column in closing_columns:
    print(f"  {column}")

print(f"\nTotal closing odds columns: {len(closing_columns)}")

# ---------------------------------------------------------
# 7. POTENTIAL LEAKAGE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("POTENTIAL DATA LEAKAGE")
print("=" * 70)

leakage_columns = [
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

for column in leakage_columns:

    if column in data.columns:
        print(f"  {column}")

# ---------------------------------------------------------
# 8. DATA TYPES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(data.dtypes.to_string())

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)