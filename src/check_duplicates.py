from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")

print("=" * 70)
print("CHECKING DUPLICATE MATCHES")
print("=" * 70)


# ---------------------------------------------------------
# LOAD ALL FILES
# ---------------------------------------------------------

files = sorted(DATA_DIR.glob("*.csv"))

dataframes = []

for file in files:

    df = pd.read_csv(file)

    # Keep track of where every row came from
    df["source_file"] = file.name

    dataframes.append(df)


data = pd.concat(
    dataframes,
    ignore_index=True,
    sort=False
)


# ---------------------------------------------------------
# CREATE DATE
# ---------------------------------------------------------

data["Date"] = pd.to_datetime(
    data["Date"],
    dayfirst=True,
    errors="coerce"
)


# ---------------------------------------------------------
# CREATE MATCH ID
# ---------------------------------------------------------

data["match_id"] = (
    data["Date"].dt.strftime("%Y-%m-%d")
    + "_"
    + data["HomeTeam"].astype(str)
    + "_"
    + data["AwayTeam"].astype(str)
)


# ---------------------------------------------------------
# FIND DUPLICATES
# ---------------------------------------------------------

duplicates = data[
    data["match_id"].duplicated(keep=False)
].copy()


duplicates = duplicates.sort_values(
    ["match_id", "source_file"]
)


print(f"\nTotal rows: {len(data):,}")

print(
    f"Unique matches: "
    f"{data['match_id'].nunique():,}"
)

print(
    f"Rows belonging to duplicated matches: "
    f"{len(duplicates):,}"
)

print(
    f"Number of duplicated match IDs: "
    f"{duplicates['match_id'].duplicated().sum():,}"
)


# ---------------------------------------------------------
# SHOW DUPLICATED MATCHES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATED MATCHES")
print("=" * 70)


display_columns = [
    "match_id",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
    "source_file",
]


existing_columns = [
    column
    for column in display_columns
    if column in duplicates.columns
]


print(
    duplicates[existing_columns]
    .to_string(index=False)
)


# ---------------------------------------------------------
# DUPLICATES BY SOURCE FILE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATES BY SOURCE FILE")
print("=" * 70)


duplicate_file_counts = (
    duplicates["source_file"]
    .value_counts()
)


print(duplicate_file_counts.to_string())


# ---------------------------------------------------------
# SAVE DUPLICATES FOR INSPECTION
# ---------------------------------------------------------

output_file = Path("data/processed/duplicate_matches.csv")

duplicates.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("DUPLICATE REPORT SAVED")
print("=" * 70)

print(f"\nFile: {output_file}")

print("\nDONE")