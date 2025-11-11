#!/usr/bin/env python

from pathlib import Path

import pandas as pd

def print_title(title):
    width = len(title) + 4
    border = '#' * width

    print(f"\n{border}")
    print(f"# {title} #")
    print(f"{border}\n")

def print_info(df):
    print(df.info())

data_dir = Path(__file__).parent.parent / 'data' / 'raw'

for file in data_dir.glob('*.csv'):
    df = pd.read_csv(file)
    print_title(file.name)
    print_info(df)
