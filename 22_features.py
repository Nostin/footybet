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

    # ========== Team-Level Contextual Features ==========
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

    # ========== 🔥 New Features ==========

    # 1. Team's average disposals to date (cumulative)
    df['team_avg_disposals'] = df.groupby('Team')['Disposals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    # Team form over last 4 games Convert result to a win=1, draw=0.5, loss=0
    result_map = {'Win': 1, 'Draw': 0.5, 'Loss': 0}
    df['result_num'] = df['Game Result'].map(result_map)

    df['team_form_last_4'] = df.groupby('Team')['result_num'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    df['avg_disposals_all'] = df.groupby('Player')['Disposals'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['form_diff'] = df['disposals_last_3'] - df['avg_disposals_all']
    df['form_diff_last_4'] = df['team_form_last_4'] - df['avg_disposals_all']

    # ========== 🔒 Safe Rolling Avg: Team Total Disposals per Game (last 4) ==========
    # Step 1: Calculate total team disposals per game (sum over all players in a match)
    team_disposals_per_game = df.groupby(['Date', 'Team'])['Disposals'].sum().reset_index()
    team_disposals_per_game.rename(columns={'Disposals': 'team_total_disposals'}, inplace=True)

    # Step 2: Sort for rolling
    team_disposals_per_game.sort_values(by=['Team', 'Date'], inplace=True)

    # Step 3: Rolling average over last 4 games, shifted to prevent leakage
    team_disposals_per_game['avg_team_disposals_last_4'] = team_disposals_per_game.groupby('Team')['team_total_disposals'] \
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())

    # Step 4: Merge it back into main DF on Date and Team
    df = df.merge(
        team_disposals_per_game[['Date', 'Team', 'avg_team_disposals_last_4']],
        on=['Date', 'Team'],
        how='left'
    )


    # ========== Targets ==========
    df['target_20'] = (df['Disposals'] >= 20).astype(int)
    df['target_25'] = (df['Disposals'] >= 25).astype(int)
    df['target_30'] = (df['Disposals'] >= 30).astype(int)

    # ========== Final Clean-up ==========
    df = df.drop_duplicates(subset=["Player", "Date"])
    df.drop(columns=['result_num'], inplace=True)
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
