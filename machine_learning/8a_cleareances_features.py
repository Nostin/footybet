from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS

engine = get_engine()


def enrich_clearances(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Player", "Date"]).reset_index(drop=True)

    for c in ["Team", "Opponent", "Venue", "Timeslot", "Conditions"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    def capped_clearances_arr(x):
        # Clearances can spike for inside mids, but huge numbers are still rare
        return np.minimum(x, 18)

    def filtered_roll_mean(g, mask_col, w, value_col):
        vals = g[value_col].where(g[mask_col])
        return vals.shift(1).rolling(w, min_periods=1).mean()

    def safe_ratio(a, b):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        out = np.where((pd.notna(b)) & (np.abs(b) > 1e-9), a / b, np.nan)
        return pd.Series(out, index=a.index if isinstance(a, pd.Series) else None)

    df["Clearances"] = pd.to_numeric(df["Clearances"], errors="coerce")
    df["clearances_capped"] = capped_clearances_arr(df["Clearances"].fillna(0).values)

    grouped_p = df.groupby("Player", sort=False)

    for w in [3, 6, 10]:
        df[f"clearances_cap_avg_last_{w}"] = grouped_p.apply(
            lambda g: g["clearances_capped"].shift(1).rolling(w, min_periods=1).mean(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"clearances_cap_med_last_{w}"] = grouped_p.apply(
            lambda g: g["clearances_capped"].shift(1).rolling(w, min_periods=1).median(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"clearances_cap_max_last_{w}"] = grouped_p.apply(
            lambda g: g["clearances_capped"].shift(1).rolling(w, min_periods=1).max(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"clearances_cap_min_last_{w}"] = grouped_p.apply(
            lambda g: g["clearances_capped"].shift(1).rolling(w, min_periods=1).min(),
            include_groups=False
        ).reset_index(level=0, drop=True)

        df[f"clearances_cap_var_last_{w}"] = grouped_p.apply(
            lambda g: g["clearances_capped"].shift(1).rolling(w, min_periods=2).var(ddof=0),
            include_groups=False
        ).reset_index(level=0, drop=True)

    df["clearances_trend_last_5"] = grouped_p["Clearances"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=True
        )
    )
    df["clearances_delta_5"] = grouped_p["Clearances"].transform(lambda x: x.shift(1) - x.shift(6))
    df["clearances_max_last_5"] = grouped_p["Clearances"].transform(lambda x: x.shift(1).rolling(5).max())
    df["clearances_min_last_5"] = grouped_p["Clearances"].transform(lambda x: x.shift(1).rolling(5).min())
    df["clearances_std_last_5"] = grouped_p["Clearances"].transform(lambda x: x.shift(1).rolling(5).std())
    df["days_since_last_game"] = grouped_p["Date"].transform(lambda x: x.diff().dt.days.shift(1))

    def is_home_game(row):
        return pd.notna(row["Venue"]) and row["Venue"] in HOME_GROUNDS.get(row["Team"], [])

    df["is_home_game"] = df.apply(is_home_game, axis=1)
    df["is_away_game"] = ~df["is_home_game"]

    cond = df["Conditions"].astype(str).str.lower() if "Conditions" in df.columns else ""
    df["is_wet_game"] = (cond == "wet")
    df["is_dry_game"] = (cond == "dry")
    df["timeslot_category"] = df["Timeslot"].astype(str).str.lower().map(
        {"day": "day", "twilight": "twilight", "night": "night"}
    ).fillna("unknown")

    grouped_p = df.groupby("Player", sort=False)

    for w in [3, 6, 10]:
        for flag, colname in [
            ("is_wet_game", f"clearances_cap_wet_avg_last_{w}"),
            ("is_dry_game", f"clearances_cap_dry_avg_last_{w}"),
            ("is_home_game", f"clearances_cap_home_avg_last_{w}"),
            ("is_away_game", f"clearances_cap_away_avg_last_{w}")
        ]:
            df[colname] = grouped_p.apply(
                lambda g: filtered_roll_mean(g, flag, w, "clearances_capped"),
                include_groups=False
            ).reset_index(level=0, drop=True)

        for ts in ["day", "night", "twilight"]:
            df[f"clearances_cap_{ts}_avg_last_{w}"] = grouped_p.apply(
                lambda g: g.assign(ts=(g["timeslot_category"] == ts))
                          .pipe(lambda gg: filtered_roll_mean(gg, "ts", w, "clearances_capped")),
                include_groups=False
            ).reset_index(level=0, drop=True)

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
        df[f"clearances_floor_score_last_{w}"] = grouped_p["Clearances"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).apply(floor_score, raw=False)
        )

    df["season_to_date_clearances_median"] = grouped_p["Clearances"].transform(
        lambda x: x.expanding().median().shift(1)
    )
    df["form_minus_season_clearances_med_last_3"] = (
        df["clearances_cap_avg_last_3"] - df["season_to_date_clearances_median"]
    )

    def merge_team_clearances_features(df_in: pd.DataFrame):
        try:
            tp = pd.read_sql(
                'SELECT "Date","Team","clearances_avg_last_5","concede_clearances_avg_last_5" '
                'FROM team_precompute',
                engine
            )
            tp["Date"] = pd.to_datetime(tp["Date"], errors="coerce")
            tp["Team"] = tp["Team"].astype(str).str.strip()
            tp = tp.dropna(subset=["Date"]).sort_values(["Team", "Date"]).reset_index(drop=True)

            tp["team_clearances_last_5_pre"] = tp.groupby("Team")["clearances_avg_last_5"].shift(1)
            tp["opp_clearances_conc_last_5_pre"] = tp.groupby("Team")["concede_clearances_avg_last_5"].shift(1)

            out = df_in.merge(
                tp[["Date", "Team", "team_clearances_last_5_pre"]]
                .rename(columns={"team_clearances_last_5_pre": "team_clearances_avg_last_5"}),
                on=["Date", "Team"], how="left"
            )

            if "Opponent" in out.columns:
                out = out.merge(
                    tp[["Date", "Team", "opp_clearances_conc_last_5_pre"]]
                    .rename(columns={
                        "Team": "Opponent",
                        "opp_clearances_conc_last_5_pre": "opp_clearances_conc_last_5"
                    }),
                    on=["Date", "Opponent"], how="left"
                )
            else:
                out["opp_clearances_conc_last_5"] = np.nan

            return out

        except Exception:
            base = df_in[["Date", "Team", "Opponent", "Clearances"]].copy()
            base["Clearances"] = base["Clearances"].fillna(0)

            team_for = (
                base.groupby(["Date", "Team"], as_index=False)["Clearances"]
                    .sum()
                    .rename(columns={"Clearances": "team_clearances"})
                    .sort_values(["Team", "Date"])
                    .reset_index(drop=True)
            )

            team_against = (
                base.groupby(["Date", "Opponent"], as_index=False)["Clearances"]
                    .sum()
                    .rename(columns={"Opponent": "Team", "Clearances": "opp_clearances_conceded"})
                    .sort_values(["Team", "Date"])
                    .reset_index(drop=True)
            )

            t = (
                team_for.merge(team_against, on=["Date", "Team"], how="left")
                        .sort_values(["Team", "Date"])
                        .reset_index(drop=True)
            )

            t["team_clearances_avg_last_5"] = (
                t.groupby("Team")["team_clearances"]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )
            t["opp_clearances_conc_last_5"] = (
                t.groupby("Team")["opp_clearances_conceded"]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )

            out = df_in.merge(
                t[["Date", "Team", "team_clearances_avg_last_5"]],
                on=["Date", "Team"], how="left"
            )

            if "Opponent" in out.columns:
                t_opp = t[["Date", "Team", "opp_clearances_conc_last_5"]].rename(columns={"Team": "Opponent"})
                out = out.merge(t_opp, on=["Date", "Opponent"], how="left")
            else:
                out["opp_clearances_conc_last_5"] = np.nan

            return out

    df = merge_team_clearances_features(df)

    def missed_time_last4(g):
        dates = g["Date"].values
        team = g["Team"].iloc[0]
        team_dates = df[df["Team"] == team][["Date"]].drop_duplicates().sort_values("Date")
        out = []
        for d in dates:
            recent_team = team_dates[team_dates["Date"] < d].tail(4)["Date"].values
            miss = False
            for td in recent_team:
                row = g[g["Date"] == td]
                if row.empty:
                    miss = True
                    break
                tog = row["Time on Ground %"].iloc[0] if "Time on Ground %" in g.columns else np.nan
                if pd.isna(tog) or tog < 50:
                    miss = True
                    break
            out.append(int(miss))
        return pd.Series(out, index=g.index)

    df["missed_time_last4"] = grouped_p.apply(
        missed_time_last4, include_groups=False
    ).reset_index(level=0, drop=True)

    df["wet_clearances_last_3"] = grouped_p.apply(
        lambda g: g["Clearances"].where(g["is_wet_game"]).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)

    df["dry_clearances_last_3"] = grouped_p.apply(
        lambda g: g["Clearances"].where(g["is_dry_game"]).shift(1).rolling(3, min_periods=1).mean(),
        include_groups=False
    ).reset_index(level=0, drop=True)

    df["wet_dry_clearances_ratio_last_3"] = safe_ratio(
        df["wet_clearances_last_3"],
        df["dry_clearances_last_3"]
    )

    df["is_home_game"] = df["is_home_game"].astype(int)
    df["is_away_game"] = df["is_away_game"].astype(int)
    df["is_wet_game"]  = df["is_wet_game"].astype(int)
    df["is_dry_game"]  = df["is_dry_game"].astype(int)

    df["target_c4"] = (df["Clearances"] >= 4).astype(int)
    df["target_c6"] = (df["Clearances"] >= 6).astype(int)
    df["target_c8"] = (df["Clearances"] >= 8).astype(int)

    df = df.drop_duplicates(subset=["Player", "Date"]).reset_index(drop=True)
    return df


train_df = pd.read_sql("SELECT * FROM player_stats_train", engine)
test_df  = pd.read_sql("SELECT * FROM player_stats_test", engine)

train_df["is_test"] = False
test_df["is_test"]  = True

combined = pd.concat([train_df, test_df], ignore_index=True)
enriched = enrich_clearances(combined)

train_enriched = enriched[~enriched["is_test"]].copy()
test_enriched  = enriched[ enriched["is_test"]].copy()

leakage_cols = [
    "Disposals", "Goals", "Behinds", "Kicks", "Handballs",
    "Game Result", "Time on Ground %", "is_test", "Clearances"
]

train_enriched = train_enriched.drop(columns=leakage_cols, errors="ignore")
test_enriched  = test_enriched.drop(columns=leakage_cols, errors="ignore")

train_enriched.to_sql("model_feed_clearances_train", con=engine, if_exists="replace", index=False)
print("✅ model_feed_clearances_train saved")

test_enriched.to_sql("model_feed_clearances_test", con=engine, if_exists="replace", index=False)
print("✅ model_feed_clearances_test saved")