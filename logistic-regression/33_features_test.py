import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from db_connect import get_engine

engine = get_engine()

HOME_GROUNDS = {
    'Adelaide': ['Adelaide'],
    'Brisbane': ['Brisbane'],
    'Carlton': ['MCG', 'Docklands'],
    'Collingwood': ['MCG'],
    'Dees': ['MCG'],
    'Essendon': ['Docklands'],
    'Fremantle': ['Perth'],
    'Geelong': ['Geelong'],
    'Suns': ['Gold Coast', 'Darwin'],
    'GWS': ['Engie', 'Canberra'],
    'Hawthorn': ['MCG', 'Launceston'],
    'Norf': ['Docklands', 'Hobart'],
    'Port': ['Adelaide'],
    'Richmond': ['MCG'],
    'Saints': ['Docklands'],
    'Sydney': ['SCG'],
    'Eagles': ['Perth'],
    'Bulldogs': ['Docklands', 'Ballarat'],
}

def enrich(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Player', 'Date'])

    # --- Helpers ---
    def capped_arr(x):
        return np.minimum(x, 35)

    def filtered_roll_mean(g, mask_col, w):
        vals = g['disp_capped'].where(g[mask_col])
        return vals.shift(1).rolling(w, min_periods=1).mean()

    # --- Base rolling windows (capped disposals) ---
    df['disp_capped'] = capped_arr(df['Disposals'].values)
    grouped_p = df.groupby('Player', sort=False)
    for w in [3, 6, 10]:
        df[f'disp_cap_avg_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_med_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).median(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_max_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).max(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_min_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=1).min(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'disp_cap_var_last_{w}'] = grouped_p.apply(
            lambda g: g['disp_capped'].shift(1).rolling(w, min_periods=2).var(ddof=0),
            include_groups=False
        ).reset_index(level=0, drop=True)

    # --- Trend / spread / rest ---
    df['disposals_trend_last_5'] = grouped_p['Disposals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df['disposals_delta_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1) - x.shift(6))
    df['disposals_max_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).max())
    df['disposals_min_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).min())
    df['disposals_std_last_5'] = grouped_p['Disposals'].transform(lambda x: x.shift(1).rolling(5).std())
    df['days_since_last_game'] = grouped_p['Date'].transform(lambda x: x.diff().dt.days.shift(1))

    # --- Home / Away flags (null safe) ---
    def is_home_game(row):
        return pd.notna(row['Venue']) and row['Venue'] in HOME_GROUNDS.get(row['Team'], [])
    df['is_home_game'] = df.apply(is_home_game, axis=1)
    df['is_away_game'] = ~df['is_home_game']

    # --- Wet / Dry / Timeslot flags ---
    cond = df['Conditions'].astype(str).str.lower()
    df['is_wet_game'] = (cond == 'wet')
    df['is_dry_game'] = (cond == 'dry')
    df['timeslot_category'] = df['Timeslot'].astype(str).str.lower().map(
        {'day': 'day', 'twilight': 'twilight', 'night': 'night'}
    ).fillna('unknown')

    # Rebuild groupby AFTER creating the flags (avoids KeyError in apply)
    grouped_p = df.groupby('Player', sort=False)

    # --- Conditional last-N windows (wet/dry/home/away/day/night/twilight), capped ---
    for w in [3, 6, 10]:
        for flag, colname in [
            ('is_wet_game', f'disp_cap_wet_avg_last_{w}'),
            ('is_dry_game', f'disp_cap_dry_avg_last_{w}'),
            ('is_home_game', f'disp_cap_home_avg_last_{w}'),
            ('is_away_game', f'disp_cap_away_avg_last_{w}')
        ]:
            df[colname] = grouped_p.apply(
                lambda g: filtered_roll_mean(g, flag, w),
                include_groups=False
            ).reset_index(level=0, drop=True)

        for ts in ['day', 'night', 'twilight']:
            df[f'disp_cap_{ts}_avg_last_{w}'] = grouped_p.apply(
                lambda g: g.assign(ts=(g['timeslot_category'] == ts))
                          .pipe(lambda gg: filtered_roll_mean(gg, 'ts', w)),
                include_groups=False
            ).reset_index(level=0, drop=True)

    # --- Floor / consistency metric on last N (median-drop score) ---
    def floor_score(series):
        if len(series) == 0 or np.all(pd.isna(series)):
            return np.nan
        med = np.nanmedian(series)
        if not np.isfinite(med) or med <= 0:
            return np.nan
        drops = med - series
        drops = drops[drops > 0]
        mean_drop = np.nanmean(drops) if len(drops) else 0.0
        return round(1 - (mean_drop / med), 3)

    for w in [3, 6, 10, 22]:
        df[f'disp_floor_score_last_{w}'] = grouped_p['Disposals'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).apply(floor_score, raw=False)
        )

    # --- Season-to-date (pre-game) medians and form vs season ---
    df['season_to_date_median'] = grouped_p['Disposals'].transform(lambda x: x.expanding().median().shift(1))
    df['form_minus_season_med_last_3'] = df['disp_cap_avg_last_3'] - df['season_to_date_median']

    # --- Team pace (pre-game) ---
    df['shifted_disposals'] = grouped_p['Disposals'].shift(1)
    team_disp = df.groupby(['Date', 'Team'], as_index=False)['shifted_disposals'].sum()
    team_disp.rename(columns={'shifted_disposals': 'team_total_disposals_pre'}, inplace=True)
    team_disp['avg_team_disposals_last_4'] = team_disp.groupby('Team')['team_total_disposals_pre'] \
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    df = df.merge(team_disp[['Date', 'Team', 'avg_team_disposals_last_4']],
                  on=['Date', 'Team'], how='left')

    # --- Opponent concessions (pre-game), guarded if Opponent column exists ---
    if 'Opponent' in df.columns:
        team_vs = df.groupby(['Date', 'Team', 'Opponent'], as_index=False)['shifted_disposals'].sum()
        # "Allowed by Team" == opponent's total disposals vs them on that date
        allowed = team_vs.rename(columns={
            'Team': 'OppFaced',
            'Opponent': 'Team',
            'shifted_disposals': 'disposals_allowed_pre'
        }).sort_values(['Team', 'Date'])
        allowed['opp_concessions_last_5'] = allowed.groupby('Team')['disposals_allowed_pre'] \
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        df = df.merge(allowed[['Date', 'Team', 'opp_concessions_last_5']],
                      on=['Date', 'Team'], how='left')
    else:
        df['opp_concessions_last_5'] = np.nan

    # --- Missed-time proxy in last 4 team games (pre-game only) ---
    def missed_time_last4(g):
        # For each row: look back at last 4 team dates and see if player absent or ToG<50
        dates = g['Date'].values
        team = g['Team'].iloc[0]
        team_dates = df[df['Team'] == team][['Date']].drop_duplicates().sort_values('Date')
        out = []
        for d in dates:
            recent_team = team_dates[team_dates['Date'] < d].tail(4)['Date'].values
            miss = False
            for td in recent_team:
                row = g[g['Date'] == td]
                if row.empty:
                    miss = True
                    break
                tog = row['Time on Ground %'].iloc[0]
                if pd.isna(tog) or tog < 50:
                    miss = True
                    break
            out.append(int(miss))
        return pd.Series(out, index=g.index)

    df['missed_time_last4'] = grouped_p.apply(missed_time_last4, include_groups=False)\
                                       .reset_index(level=0, drop=True)

    # --- Wet/Dry rolling avgs (uncapped; you also have capped versions above) ---
    df['wet_disposals_last_3'] = grouped_p.apply(
        lambda g: g['Disposals'].where(g['is_wet_game']).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df['dry_disposals_last_3'] = grouped_p.apply(
        lambda g: g['Disposals'].where(g['is_dry_game']).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df['wet_dry_disp_ratio_last_3'] = df['wet_disposals_last_3'] / df['dry_disposals_last_3']

    # --- One-hot-ish flags (ints) ---
    df['is_home_game'] = df['is_home_game'].astype(int)
    df['is_away_game'] = df['is_away_game'].astype(int)
    df['is_wet_game'] = df['is_wet_game'].astype(int)
    df['is_dry_game'] = df['is_dry_game'].astype(int)

    # --- Targets (binary) ---
    df['target_20'] = (df['Disposals'] >= 20).astype(int)
    df['target_25'] = (df['Disposals'] >= 25).astype(int)
    df['target_30'] = (df['Disposals'] >= 30).astype(int)

    # --- Clean-up ---
    df = df.drop(columns=['shifted_disposals'], errors='ignore')
    df = df.drop_duplicates(subset=['Player', 'Date'])

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
