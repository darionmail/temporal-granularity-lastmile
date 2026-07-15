"""
01_preprocessing.py

Ucitava finalni Zagreb-zapad listopad uzorak, projicira geografske koordinate
u lokalni metricki sustav (x, y u metrima), i sprema obogaceni dataset
spreman za sve tri optimizacijske metode.

Pokreni: python3 01_preprocessing.py
"""
import sys
import os
import pandas as pd
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from geo_utils import latlon_to_xy

INPUT_FILE = config["data"]["input_file"]
OUTPUT_FILE = config["data"]["input_file"]


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    df["EventDatetime"] = pd.to_datetime(df["EventDatetime"], errors="coerce")
    df["date"] = df["EventDatetime"].dt.date

    # EventGeoX = longitude (lambda), EventGeoY = latitude (phi)
    x, y = latlon_to_xy(df["EventGeoY"].values, df["EventGeoX"].values)
    df["x_m"] = x
    df["y_m"] = y

    # Sortiraj po danu i kuriru za konzistentnost
    df = df.sort_values(["date", "UserID"]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("=" * 60)
    print("PREPROCESSING ZAVRSEN")
    print("=" * 60)
    print(f"Ukupno redova: {len(df):,}")
    print(f"Broj kurira: {df['UserID'].nunique()}")
    print(f"Broj dana: {df['date'].nunique()}")
    print(f"\nRaspon x (m): {df['x_m'].min():.0f} do {df['x_m'].max():.0f}")
    print(f"Raspon y (m): {df['y_m'].min():.0f} do {df['y_m'].max():.0f}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
