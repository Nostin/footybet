import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from db_connect import get_engine

engine = get_engine()

def enrich(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Player', 'Date'], inplace=True)

    grouped = df.groupby('Player')
    rolling_windows = [3, 5]
    rolling_cols = ['Disposals', 'Kicks', 'Handballs', 'Marks', 'Tackles', 'Clearances', 'Time on Ground %']

    for col in rolling_cols:
        for window in rolling_windows:
            safe_col = col.lower().replace(" ", "_")
            if col == "Disposals" and window == 5:
                continue
            df[f'{safe_col}_last_{window}'] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    df['disposals_trend_last_5'] = grouped['Disposals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df['disposals_delta_5'] = grouped['Disposals'].transform(lambda x: x.shift(1) - x.shift(6))
    df['disposals_max_last_5'] = grouped['Disposals'].transform(lambda x: x.shift(1).rolling(5).max())
    df['disposals_min_last_5'] = grouped['Disposals'].transform(lambda x: x.shift(1).rolling(5).min())
    df['disposals_std_last_5'] = grouped['Disposals'].transform(lambda x: x.shift(1).rolling(5).std())
    df['days_since_last_game'] = grouped['Date'].transform(lambda x: x.diff().dt.days.shift(1))

    home_ground_map = {
        'Hawthorn': 'MCG', 'Collingwood': 'MCG', 'Carlton': 'MCG', 'Essendon': 'MCG',
        'Dees': 'MCG', 'Richmond': 'MCG', 'Saints': 'Docklands', 'Bulldogs': 'Docklands',
        'Norf': 'Docklands', 'Sydney': 'SCG', 'GWS': 'Engie', 'Brisbane': 'Brisbane',
        'Suns': 'Gold Coast', 'Eagles': 'Perth', 'Fremantle': 'Perth', 'Adelaide': 'Adelaide',
        'Port': 'Adelaide', 'Geelong': 'Geelong'
    }

    df['is_home_game'] = df.apply(lambda row: row['Venue'] == home_ground_map.get(row['Team'], ''), axis=1)
    df['is_away_game'] = ~df['is_home_game']

    # --- WET/DRY Weather Features ---
    df['is_wet_game'] = df['Conditions'].str.lower() == 'wet'
    df['is_dry_game'] = df['Conditions'].str.lower() == 'dry'

    # Rolling average disposals in last 3 wet games for each player
    df['wet_disposals_last_3'] = (
        grouped.apply(lambda group: 
            group['Disposals']
            .where(group['is_wet_game'])
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
        ).reset_index(level=0, drop=True)
    )

    # Rolling average disposals in last 3 dry games for each player
    df['dry_disposals_last_3'] = (
        grouped.apply(lambda group: 
            group['Disposals']
            .where(group['is_dry_game'])
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
        ).reset_index(level=0, drop=True)
    )

    # Wet vs Dry disposals ratio (last 3 games of each, shift to avoid leakage)
    df['wet_dry_disp_ratio_last_3'] = (
        df['wet_disposals_last_3'] / df['dry_disposals_last_3']
    )

    # Boolean: is this game wet?
    df['is_wet_game'] = df['is_wet_game'].astype(int)
    df['is_dry_game'] = df['is_dry_game'].astype(int)

    df['timeslot_category'] = df['Timeslot'].str.lower().map({
        'day': 'day', 'twilight': 'twilight', 'night': 'night'
    }).fillna('unknown')

    df['opp_avg_disposals_allowed_last_3'] = df.get('opp_avg_disposals_allowed_last_3', np.nan)

    df['avg_disposals_all'] = grouped['Disposals'].transform(lambda x: x.expanding().mean().shift(1))
    df['shifted_disposals'] = grouped['Disposals'].shift(1)

    team_disposals = df.groupby(['Date', 'Team'])['shifted_disposals'].sum().reset_index()
    team_disposals.rename(columns={'shifted_disposals': 'team_total_disposals'}, inplace=True)

    team_disposals['avg_team_disposals_last_4'] = team_disposals.groupby('Team')['team_total_disposals'] \
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())

    df = df.merge(
        team_disposals[['Date', 'Team', 'avg_team_disposals_last_4']],
        on=['Date', 'Team'],
        how='left'
    )

    df['target_20'] = (df['Disposals'] >= 20).astype(int)
    df['target_25'] = (df['Disposals'] >= 25).astype(int)
    df['target_30'] = (df['Disposals'] >= 30).astype(int)

    df = df.drop(columns=['shifted_disposals'], errors='ignore')
    df = df.drop_duplicates(subset=["Player", "Date"])

    return df

# 👇 Combine before enriching
train_df = pd.read_sql('SELECT * FROM player_stats_train', engine)
test_df = pd.read_sql('SELECT * FROM player_stats_test', engine)
test_df['is_test'] = True
train_df['is_test'] = False

combined = pd.concat([train_df, test_df], ignore_index=True)
enriched = enrich(combined)

train_enriched = enriched[enriched['is_test'] == False].copy()
test_enriched = enriched[enriched['is_test'] == True].copy()

# Drop leakage fields
leakage_cols = ['Disposals', 'Goals', 'Behinds', 'Kicks', 'Handballs', 'Game Result', 'Time on Ground %', 'is_test']
train_enriched = train_enriched.drop(columns=leakage_cols, errors='ignore')
test_enriched = test_enriched.drop(columns=leakage_cols, errors='ignore')

# Save
train_enriched.to_sql("model_feed_train", con=engine, if_exists="replace", index=False)
print("✅ model_feed_train saved")

test_enriched.to_sql("model_feed_test", con=engine, if_exists="replace", index=False)
print("✅ model_feed_test saved")
