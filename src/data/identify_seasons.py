from pathlib import Path
import pandas as pd

from src.utils import config


DATA_DIR = config.RAW_DIR

print("=" * 80)
print("IDENTIFYING CSV FILES AND SEASONS")
print("=" * 80)


files = sorted(DATA_DIR.glob("*.csv"))


for file in files:

    df = pd.read_csv(file)

    print("\n" + "-" * 80)
    print(f"FILE: {file.name}")
    print("-" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # DATE INFORMATION
    # -----------------------------------------------------

    if "Date" in df.columns:

        dates = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

        valid_dates = dates.dropna()

        print(f"Valid dates: {len(valid_dates):,}")

        if len(valid_dates) > 0:

            print(
                f"Earliest date: {valid_dates.min().date()}"
            )

            print(
                f"Latest date:   {valid_dates.max().date()}"
            )

        else:
            print("No valid dates found.")

    # -----------------------------------------------------
    # TEAMS
    # -----------------------------------------------------

    if "HomeTeam" in df.columns:

        teams = set(
            df["HomeTeam"].dropna()
        ) | set(
            df["AwayTeam"].dropna()
        )

        print(f"Teams: {len(teams)}")

        print(
            ", ".join(sorted(teams))
        )

    # -----------------------------------------------------
    # FIRST 5 MATCHES
    # -----------------------------------------------------

    print("\nFirst 5 matches:")

    columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR"
    ]

    available = [
        c for c in columns
        if c in df.columns
    ]

    print(
        df[available]
        .head(5)
        .to_string(index=False)
    )


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)