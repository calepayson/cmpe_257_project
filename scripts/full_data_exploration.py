#!/usr/bin/env python

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def print_title(title: str):
    width = len(title) + 4
    print("\n" + ("#" * width))
    print(f"# {title} #")
    print("#" * width)


def main():
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    plot_dir = Path(__file__).parent.parent / "experiments" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prices_csv = raw_dir / "prices.csv"

    df = pd.read_csv(prices_csv)

    print("Let's see what we're working with")

    print_title("Head")
    print(df.head())

    print_title("Info")
    print("  - Two non-numeric types: date and symbol")
    print("  - No nulls!")
    print(df.info())

    print_title("Symbol Exploration")
    print("  - About 500 different symbols")
    print("  - Different number of records for each symbol")
    print(df["symbol"].value_counts())

    print_title("Description")
    print("  - Looks like the data will have a pretty long tail")
    print(df.describe())

    print_title("Histogram")
    histogram_path = plot_dir / "prices_histogram.png"
    print(f"  - Saving histogram to {histogram_path}")
    print("  - Every variable's histogram is massively skewed right")
    print("  - Besides that each distribution seems standard")
    print("  - The volume distribution is on a different scale than the rest")
    df.hist(bins=50, figsize=(12, 8))
    plt.savefig(histogram_path)


if __name__ == "__main__":
    main()
