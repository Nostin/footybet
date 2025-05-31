import pandas as pd
from sqlalchemy import create_engine
from db_connect import get_engine

# Connect to your database
engine = get_engine()

def enrich(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Player', 'Date'], inplace=True)

    # ========== Player-Level Rolling Stats ==========
    grouped = df.groupby('Player')
    rolling_windows = [3, 5]
    rolling_cols = ['Disposals', 'Kicks', 'Handballs', 'Marks', 'Tackles', 'Clearances', 'Time on Ground %']

    for col in rolling_cols:
        for window in rolling_windows:
            safe_col = col.lower().replace(" ", "_")
            df[f'{safe_col}_last_{window}'] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    df['disposals_std_last_5'] = grouped['Disposals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )
    df['days_since_last_game'] = grouped['Date'].transform(
        lambda x: x.diff().dt.days.shift(1)
    )

    # ========== Contextual Features ==========
    home_ground_map = {
        'Hawthorn': 'MCG', 'Collingwood': 'MCG', 'Carlton': 'MCG', 'Essendon': 'MCG',
        'Dees': 'MCG', 'Richmond': 'MCG', 'Saints': 'Docklands', 'Bulldogs': 'Docklands',
        'Norf': 'Docklands', 'Sydney': 'SCG', 'GWS': 'Engie', 'Brisbane': 'Brisbane',
        'Suns': 'Gold Coast', 'Eagles': 'Perth', 'Fremantle': 'Perth', 'Adelaide': 'Adelaide',
        'Port': 'Adelaide', 'Geelong': 'Geelong'
    }

    df['is_home_game'] = df.apply(lambda row: row['Venue'] == home_ground_map.get(row['Team'], ''), axis=1)
    df['is_away_game'] = ~df['is_home_game']

    df['timeslot_category'] = df['Timeslot'].str.lower().map({
        'day': 'day', 'twilight': 'twilight', 'night': 'night'
    }).fillna('unknown')

    # ========== Opponent Team Disposal Tendencies ==========
    team_totals = df.groupby(['Date', 'Team'])['Disposals'].sum().reset_index()
    team_totals.rename(columns={'Disposals': 'team_disposals'}, inplace=True)

    df = df.merge(team_totals, on=['Date', 'Team'], how='left')

    opp_df = df[['Date', 'Opponent', 'team_disposals']].rename(columns={
        'Opponent': 'Team', 'team_disposals': 'disposals_conceded'
    })
    opp_df = opp_df.drop_duplicates(subset=['Date', 'Team'])
    opp_df.sort_values(by=['Team', 'Date'], inplace=True)
    opp_df['opp_avg_disposals_allowed_last_3'] = opp_df.groupby('Team')['disposals_conceded'] \
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    df = df.merge(
        opp_df[['Date', 'Team', 'opp_avg_disposals_allowed_last_3']],
        left_on=['Date', 'Opponent'],
        right_on=['Date', 'Team'],
        how='left',
        suffixes=('', '_opponent')
    ).drop(columns=['Team_opponent'])

    # ========== Targets ==========
    df['target_20'] = (df['Disposals'] >= 20).astype(int)
    df['target_25'] = (df['Disposals'] >= 25).astype(int)
    df['target_30'] = (df['Disposals'] >= 30).astype(int)

    # ========== Final Clean-up ==========
    df = df.drop_duplicates(subset=["Player", "Date"])
    return df

# Load and enrich train data
train_df = pd.read_sql('SELECT * FROM player_stats_train', engine)
train_enriched = enrich(train_df)
train_enriched.to_sql("model_feed_train", con=engine, if_exists="replace", index=False)
print("✅ model_feed_train saved")

# Load and enrich test data
test_df = pd.read_sql('SELECT * FROM player_stats_test', engine)
test_enriched = enrich(test_df)
test_enriched.to_sql("model_feed_test", con=engine, if_exists="replace", index=False)
print("✅ model_feed_test saved")
