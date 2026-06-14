import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs('data/processed', exist_ok=True)

def build_features():
    # ── Load merged data ─────────────────────────────────
    df = pd.read_csv('data/processed/merged.csv')
    print(f"Loaded: {df.shape}")

    # ── Step 1: dropping useless columns ────────────────
    drop_cols = [
        'resultId', 'raceId', 'driverId', 'constructorId',
        'statusId', 'number', 'position', 'positionText',
        'time', 'milliseconds', 'fastestLap', 'rank',
        'fastestLapTime', 'fastestLapSpeed',
        'forename', 'surname',
        'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time',
        'fp3_date', 'fp3_time', 'quali_date', 'quali_time',
        'sprint_date', 'sprint_time', 'url',
        'positions_gained'
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"After dropping cols: {df.shape}")

    # ── Step 2: DNF flag  ───────────────────────────
    df['is_dnf'] = df['status'].apply(
        lambda x: 0 if x == 'Finished' else 1
    )
  

    # ── Step 3: Rolling driver form ──────────────────────
    df = df.sort_values(['code', 'year', 'round'])
    df['driver_form'] = df.groupby('code')['positionOrder']\
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
  

    # ── Step 4: Encode categorical columns ───────────────
    le = LabelEncoder()
    df['driver_enc']  = le.fit_transform(df['code'].astype(str))
    df['team_enc']    = le.fit_transform(df['name_team'].astype(str))
    df['circuit_enc'] = le.fit_transform(df['name_circuit'].astype(str))
    df['nation_enc']  = le.fit_transform(df['nationality'].astype(str))
    

    # ── Step 5: Final features selection ───────────────
    feature_cols = [
        'year', 'round', 'circuit_enc',
        'driver_enc', 'team_enc', 'nation_enc',
        'grid', 'quali_position',
        'driver_pts_before', 'driver_standing_before',
        'team_pts_before', 'team_standing_before',
        'pit_stop_count', 'driver_form', 'is_dnf',
        'laps', 'points',
        'positionOrder'  # target
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_final = df[feature_cols].copy()

    # ── Step 5.5: Remaining nulls filling───────────────
    df_final['pit_stop_count'] = df_final['pit_stop_count'].fillna(0)

    # driver_form
    df_final['driver_form'] = df_final['driver_form'].fillna(
        df_final['grid']
    )

    # ── Step 6: dropping remaining nulls ────────────────
    before = len(df_final)
    df_final = df_final.dropna()
    after = len(df_final)
    print(f"Dropped {before - after} null rows")

    # ── Step 7: Saving ─────────────────────────────────
    df_final.to_csv('data/processed/features.csv', index=False)
    print(f"Final shape  : {df_final.shape}")
    print(f"Columns      : {df_final.columns.tolist()}")

if __name__ == "__main__":
    build_features()