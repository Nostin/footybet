from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS

engine = get_engine()

def enrich_goals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values(['Player', 'Date']).reset_index(drop=True)

    # Clean strings
    for c in ['Team', 'Opponent', 'Venue', 'Timeslot', 'Conditions']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # --- Helpers ---
    def capped_goals_arr(x):
        # Goals have long tails for KPFs; cap to stabilize variance
        return np.minimum(x, 6)

    def filtered_roll_mean(g, mask_col, w, value_col):
        vals = g[value_col].where(g[mask_col])
        return vals.shift(1).rolling(w, min_periods=1).mean()

    # --- Base rolling windows (capped goals) ---
    df['goals_capped'] = capped_goals_arr(df['Goals'].fillna(0).values)
    df['shots'] = (df['Goals'].fillna(0) + df['Behinds'].fillna(0)).astype(float)
    df['accuracy'] = np.where(df['shots'] > 0, df['Goals'] / df['shots'], np.nan)

    grouped_p = df.groupby('Player', sort=False)

    for w in [3, 6, 10]:
        # Goals level
        df[f'goals_cap_avg_last_{w}'] = grouped_p.apply(
            lambda g: g['goals_capped'].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'goals_cap_med_last_{w}'] = grouped_p.apply(
            lambda g: g['goals_capped'].shift(1).rolling(w, min_periods=1).median(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'goals_cap_max_last_{w}'] = grouped_p.apply(
            lambda g: g['goals_capped'].shift(1).rolling(w, min_periods=1).max(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'goals_cap_min_last_{w}'] = grouped_p.apply(
            lambda g: g['goals_capped'].shift(1).rolling(w, min_periods=1).min(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'goals_cap_var_last_{w}'] = grouped_p.apply(
            lambda g: g['goals_capped'].shift(1).rolling(w, min_periods=2).var(ddof=0),
            include_groups=False
        ).reset_index(level=0, drop=True)

        # Shot volume & accuracy
        df[f'shots_avg_last_{w}'] = grouped_p.apply(
            lambda g: g['shots'].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)
        df[f'accuracy_avg_last_{w}'] = grouped_p.apply(
            lambda g: g['accuracy'].shift(1).rolling(w, min_periods=2).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)

    # --- Trend / spread / rest (Goals) ---
    df['goals_trend_last_5'] = grouped_p['Goals'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df['goals_delta_5'] = grouped_p['Goals'].transform(lambda x: x.shift(1) - x.shift(6))
    df['goals_max_last_5'] = grouped_p['Goals'].transform(lambda x: x.shift(1).rolling(5).max())
    df['goals_min_last_5'] = grouped_p['Goals'].transform(lambda x: x.shift(1).rolling(5).min())
    df['goals_std_last_5'] = grouped_p['Goals'].transform(lambda x: x.shift(1).rolling(5).std())
    df['days_since_last_game'] = grouped_p['Date'].transform(lambda x: x.diff().dt.days.shift(1))

    # --- Home / Away flags ---
    def is_home_game(row):
        return pd.notna(row['Venue']) and row['Venue'] in HOME_GROUNDS.get(row['Team'], [])
    df['is_home_game'] = df.apply(is_home_game, axis=1)
    df['is_away_game'] = ~df['is_home_game']

    # --- Wet / Dry / Timeslot flags ---
    cond = df['Conditions'].astype(str).str.lower() if 'Conditions' in df.columns else ''
    df['is_wet_game'] = (cond == 'wet')
    df['is_dry_game'] = (cond == 'dry')
    df['timeslot_category'] = df['Timeslot'].astype(str).str.lower().map(
        {'day': 'day', 'twilight': 'twilight', 'night': 'night'}
    ).fillna('unknown')

    grouped_p = df.groupby('Player', sort=False)

    # --- Conditional last-N windows (by context) on capped goals ---
    for w in [3, 6, 10]:
        for flag, colname in [
            ('is_wet_game', f'goals_cap_wet_avg_last_{w}'),
            ('is_dry_game', f'goals_cap_dry_avg_last_{w}'),
            ('is_home_game', f'goals_cap_home_avg_last_{w}'),
            ('is_away_game', f'goals_cap_away_avg_last_{w}')
        ]:
            df[colname] = grouped_p.apply(
                lambda g: filtered_roll_mean(g, flag, w, 'goals_capped'),
                include_groups=False
            ).reset_index(level=0, drop=True)

        for ts in ['day', 'night', 'twilight']:
            df[f'goals_cap_{ts}_avg_last_{w}'] = grouped_p.apply(
                lambda g: g.assign(ts=(g['timeslot_category'] == ts))
                          .pipe(lambda gg: filtered_roll_mean(gg, 'ts', w, 'goals_capped')),
                include_groups=False
            ).reset_index(level=0, drop=True)

    # --- Floor / consistency metric on last N (median-drop score) over GOALS ---
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
        df[f'goals_floor_score_last_{w}'] = grouped_p['Goals'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).apply(floor_score, raw=False)
        )
        df[f'accuracy_floor_score_last_{w}'] = grouped_p['accuracy'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=3).apply(floor_score, raw=False)
        )

    # --- Season-to-date medians (pre-game) + form deltas ---
    df['season_to_date_goals_median'] = grouped_p['Goals'].transform(lambda x: x.expanding().median().shift(1))
    df['season_to_date_shots_median'] = grouped_p['shots'].transform(lambda x: x.expanding().median().shift(1))
    df['season_to_date_acc_median']   = grouped_p['accuracy'].transform(lambda x: x.expanding().median().shift(1))

    df['form_minus_season_goals_med_last_3'] = df['goals_cap_avg_last_3'] - df['season_to_date_goals_median']
    df['form_minus_season_acc_med_last_3']   = df['accuracy_avg_last_3'] - df['season_to_date_acc_median']

    # --- Team attack pace & opponent concessions (prefer team_precompute; fallback if missing) ---
    def merge_team_goals_features(df_in: pd.DataFrame):
        # Try to read from team_precompute if present
        try:
            tp = pd.read_sql(
                'SELECT "Date","Team","goals_for_avg_last_5","concede_goals_avg_last_5" '
                'FROM team_precompute',
                engine
            )
            tp["Date"] = pd.to_datetime(tp["Date"], errors="coerce")
            tp["Team"] = tp["Team"].astype(str).str.strip()
            tp = tp.dropna(subset=["Date"]).sort_values(["Team", "Date"]).reset_index(drop=True)

            # make pre-game (shift by team)
            tp["team_goals_last_5_pre"] = tp.groupby("Team")["goals_for_avg_last_5"].shift(1)
            tp["opp_goals_conc_last_5_pre"] = tp.groupby("Team")["concede_goals_avg_last_5"].shift(1)

            out = df_in.merge(
                tp[["Date","Team","team_goals_last_5_pre"]]
                .rename(columns={"team_goals_last_5_pre": "team_goals_avg_last_5"}),
                on=["Date","Team"], how="left"
            )
            if "Opponent" in out.columns:
                out = out.merge(
                    tp[["Date","Team","opp_goals_conc_last_5_pre"]]
                    .rename(columns={"Team":"Opponent",
                                    "opp_goals_conc_last_5_pre":"opp_goals_conc_last_5"}),
                    on=["Date","Opponent"], how="left"
                )
            else:
                out["opp_goals_conc_last_5"] = np.nan
            return out

        except Exception:
            # ---------- Fallback: build from the current dataframe ----------
            base = df_in[['Date','Team','Opponent','Goals']].copy()
            base['Goals'] = base['Goals'].fillna(0)

            # Team goals per (Date, Team)
            team_for = (
                base.groupby(['Date','Team'], as_index=False)['Goals']
                    .sum()
                    .rename(columns={'Goals': 'team_goals'})
                    .sort_values(['Team','Date'])
                    .reset_index(drop=True)
            )

            # Goals conceded per (Date, Team) = sum of opponents' goals
            team_against = (
                base.groupby(['Date','Opponent'], as_index=False)['Goals']
                    .sum()
                    .rename(columns={'Opponent': 'Team', 'Goals': 'opp_goals_conceded'})
                    .sort_values(['Team','Date'])
                    .reset_index(drop=True)
            )

            # Join the two aggregates
            t = (team_for
                .merge(team_against, on=['Date','Team'], how='left')
                .sort_values(['Team','Date'])
                .reset_index(drop=True)
            )

            # IMPORTANT: use transform so the result aligns to t.index
            t['team_goals_avg_last_5'] = (
                t.groupby('Team')['team_goals']
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )
            t['opp_goals_conc_last_5'] = (
                t.groupby('Team')['opp_goals_conceded']
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )

            # Merge back to player rows
            out = df_in.merge(
                t[['Date','Team','team_goals_avg_last_5']],
                on=['Date','Team'], how='left'
            )
            if 'Opponent' in out.columns:
                t_opp = t[['Date','Team','opp_goals_conc_last_5']].rename(columns={'Team':'Opponent'})
                out = out.merge(t_opp, on=['Date','Opponent'], how='left')
            else:
                out['opp_goals_conc_last_5'] = np.nan

            return out

    df = merge_team_goals_features(df)

    # --- Missed-time proxy (same as your disposals version) ---
    def missed_time_last4(g):
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
                tog = row['Time on Ground %'].iloc[0] if 'Time on Ground %' in g.columns else np.nan
                if pd.isna(tog) or tog < 50:
                    miss = True
                    break
            out.append(int(miss))
        return pd.Series(out, index=g.index)

    df['missed_time_last4'] = grouped_p.apply(missed_time_last4, include_groups=False)\
                                       .reset_index(level=0, drop=True)

    # --- Wet/Dry rolling avgs (goals) ---
    df['wet_goals_last_3'] = grouped_p.apply(
        lambda g: g['Goals'].where(g['is_wet_game']).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df['dry_goals_last_3'] = grouped_p.apply(
        lambda g: g['Goals'].where(g['is_dry_game']).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)
    df['wet_dry_goals_ratio_last_3'] = df['wet_goals_last_3'] / df['dry_goals_last_3']

    # --- One-hot-ish ints ---
    df['is_home_game'] = df['is_home_game'].astype(int)
    df['is_away_game'] = df['is_away_game'].astype(int)
    df['is_wet_game']  = df['is_wet_game'].astype(int)
    df['is_dry_game']  = df['is_dry_game'].astype(int)

    # --- Targets (binary for 1+, 2+, 3+) ---
    df['target_g1'] = (df['Goals'] >= 1).astype(int)
    df['target_g2'] = (df['Goals'] >= 2).astype(int)
    df['target_g3'] = (df['Goals'] >= 3).astype(int)

    # --- Clean-up / de-dup ---
    df = df.drop_duplicates(subset=['Player', 'Date']).reset_index(drop=True)
    return df


# ===== Combine before enriching (same pattern as your disposals pipeline) =====
train_df = pd.read_sql('SELECT * FROM player_stats_train', engine)
test_df  = pd.read_sql('SELECT * FROM player_stats_test',  engine)
train_df['is_test'] = False
test_df['is_test']  = True

combined  = pd.concat([train_df, test_df], ignore_index=True)
enriched  = enrich_goals(combined)

train_enriched = enriched[~enriched['is_test']].copy()
test_enriched  = enriched[ enriched['is_test']].copy()

# Drop leakage fields (post-enrichment); keep only pre-game features + targets
leakage_cols = ['Disposals', 'Goals', 'Behinds', 'Kicks', 'Handballs',
                'Game Result', 'Time on Ground %', 'is_test']
train_enriched = train_enriched.drop(columns=leakage_cols, errors='ignore')
test_enriched  = test_enriched.drop(columns=leakage_cols,  errors='ignore')

# Save
train_enriched.to_sql("model_feed_goals_train", con=engine, if_exists="replace", index=False)
print("✅ model_feed_goals_train saved")

test_enriched.to_sql("model_feed_goals_test", con=engine, if_exists="replace", index=False)
print("✅ model_feed_goals_test saved")
