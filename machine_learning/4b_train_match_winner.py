# machine_learning/4_train_match_winner.py
from pathlib import Path
import sys, os, json, argparse
import numpy as np
import pandas as pd
from sqlalchemy import text as sqtext
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, accuracy_score
)
from sklearn.inspection import permutation_importance
import joblib

# ---------- Project root / imports ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_connect import get_engine
from util import HOME_GROUNDS  # expects your canonical team keys

engine = get_engine()

MODELS_DIR = (ROOT / "machine_learning" / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "team_winner_logreg.pkl"
META_PATH  = MODELS_DIR / "team_winner_meta.json"
IMP_CSV    = MODELS_DIR / "team_winner_feature_importance.csv"
RELIAB_CSV = MODELS_DIR / "team_winner_reliability.csv"

# ---------- Args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--use-team-precompute", action="store_true", default=True,
                help="Use team_precompute features if available (use --stats-mode pre when building it).")
ap.add_argument("--exclude-finals", action="store_true", default=True,
                help="Exclude finals from training labels.")
ap.add_argument("--n-splits", type=int, default=5)
ap.add_argument("--perm-importance", action="store_true", default=False,
                help="Compute permutation importance on the last validation fold (slower).")
args = ap.parse_args()

# ---------- Helpers ----------
FINALS = {"EF","QF","SF","PF","GF"}
def is_finals_round(x):
    if x is None: return False
    return str(x).strip().upper() in FINALS

def is_home(team: str, venue: str) -> int:
    if team is None or venue is None:
        return 0
    return int(str(venue).strip() in HOME_GROUNDS.get(str(team).strip(), []))

def timeslot_dummies(ts: str):
    ts = (ts or "").strip().lower()
    return {
        "ts_day":       1 if ts == "day" else 0,
        "ts_night":     1 if ts == "night" else 0,
        "ts_twilight":  1 if ts == "twilight" else 0,
        "ts_unknown":   1 if ts not in ("day", "night", "twilight") else 0,
    }

# ---------- Load per-team rows and pair into matches ----------
tg = pd.read_sql('SELECT * FROM team_games', engine)
tg["Date"] = pd.to_datetime(tg["Date"], errors="coerce")
tg = tg.dropna(subset=["Date"]).copy()

need = ["Team","Opponent","Venue","Timeslot",
        "elo_pre","g2_rating_pre","g2_rd_pre","g2_vol_pre",
        "points_for","points_against","Round"]
tg = tg.dropna(subset=[c for c in need if c in tg.columns])

pair_key = (
    tg["Date"].dt.strftime("%Y-%m-%d") + "|" +
    tg[["Team","Opponent"]].apply(lambda x: "|".join(sorted(x)), axis=1)
)
tg["_pair_key"] = pair_key

pairs = []
for _, grp in tg.groupby("_pair_key"):
    if len(grp) != 2:
        continue
    g = grp.sort_values(["Team","Opponent"]).reset_index(drop=True)
    a, b = g.iloc[0], g.iloc[1]

    # label (drop draws)
    if any(pd.isna([a["points_for"], a["points_against"], b["points_for"], b["points_against"]])):
        continue
    if int(a["points_for"]) == int(b["points_for"]):
        continue

    y = 1 if a["points_for"] > a["points_against"] else 0

    feat = {}
    feat["elo_diff"]     = float(a["elo_pre"])       - float(b["elo_pre"])
    feat["glicko_diff"]  = float(a["g2_rating_pre"]) - float(b["g2_rating_pre"])
    feat["rd_diff"]      = float(a["g2_rd_pre"])     - float(b["g2_rd_pre"])
    feat["vol_diff"]     = float(a["g2_vol_pre"])    - float(b["g2_vol_pre"])
    # home flags → edge
    hA, hB = is_home(a["Team"], a["Venue"]), is_home(b["Team"], b["Venue"])
    feat["home_edge"] = hA - hB  # 1 if A home, -1 if B home, 0 neutral
    # timeslot one-hots
    feat.update(timeslot_dummies(a.get("Timeslot", "")))
    # optional: Elo expectation
    feat["elo_exp"]      = 1.0 / (1.0 + 10.0 ** (-(float(a["elo_pre"]) - float(b["elo_pre"])) / 400.0))

    pairs.append({
        "Date": a["Date"], "Round": a.get("Round"),
        "team_a": a["Team"], "team_b": b["Team"],
        "Venue": a["Venue"], "Timeslot": a["Timeslot"],
        **feat, "y": y
    })

Xy = pd.DataFrame(pairs).sort_values("Date").reset_index(drop=True)
if args.exclude_finals:
    Xy = Xy[~Xy["Round"].apply(is_finals_round)].reset_index(drop=True)

if Xy.empty:
    raise RuntimeError("No training rows assembled from team_games.")

# ---------- Base features ----------
feature_cols = [
    "elo_diff","glicko_diff","rd_diff","vol_diff",
    "home_edge","ts_day","ts_night","ts_twilight","ts_unknown",
    "elo_exp"
]

# ---------- Add team_precompute diffs, matchup, and NEW interstate edges ----------
def add_team_precompute_features(Xy: pd.DataFrame, feature_cols: list):
    if not args.use_team_precompute:
        return Xy, feature_cols

    try:
        sample = pd.read_sql('SELECT * FROM team_precompute LIMIT 0', engine)
    except Exception:
        return Xy, feature_cols

    available = set(sample.columns)
    FEAT_BASE = [
        "score_avg_last_5","inside50_avg_last_5","turnovers_avg_last_5","free_kicks_avg_last_5",
        "disposals_avg_last_5","clearances_avg_last_5","tackles_avg_last_5","marks_avg_last_5",
        "concede_disposals_avg_last_5","concede_tackles_avg_last_5","concede_marks_avg_last_5","concede_clearances_avg_last_5",
    ]
    # NEW: travel/secondary-home flags
    EDGE_BIN = ["is_interstate", "is_secondary_home"]

    use_cols = [c for c in FEAT_BASE if c in available]
    edge_cols = [c for c in EDGE_BIN if c in available]
    if not use_cols and not edge_cols:
        return Xy, feature_cols

    sel_cols = ['"Date"','"Team"'] + [f'"{c}"' for c in (use_cols + edge_cols)]
    tp = pd.read_sql(f"SELECT {','.join(sel_cols)} FROM team_precompute", engine)
    tp["Date"] = pd.to_datetime(tp["Date"], errors="coerce")
    tp = tp.dropna(subset=["Date"]).copy()

    # Merge A & B snapshots
    if use_cols:
        Xy = Xy.merge(tp[["Date","Team"] + use_cols].rename(columns={c: f"A_{c}" for c in use_cols}),
                      left_on=["Date","team_a"], right_on=["Date","Team"], how="left").drop(columns=["Team"])
        Xy = Xy.merge(tp[["Date","Team"] + use_cols].rename(columns={c: f"B_{c}" for c in use_cols}),
                      left_on=["Date","team_b"], right_on=["Date","Team"], how="left").drop(columns=["Team"])

    if edge_cols:
        Xy = Xy.merge(tp[["Date","Team"] + edge_cols].rename(columns={c: f"A_{c}" for c in edge_cols}),
                      left_on=["Date","team_a"], right_on=["Date","Team"], how="left").drop(columns=["Team"])
        Xy = Xy.merge(tp[["Date","Team"] + edge_cols].rename(columns={c: f"B_{c}" for c in edge_cols}),
                      left_on=["Date","team_b"], right_on=["Date","Team"], how="left").drop(columns=["Team"])

    # Diffs (A - B) from rolling means
    for name in [
        "score_avg_last_5","inside50_avg_last_5","turnovers_avg_last_5","free_kicks_avg_last_5",
        "disposals_avg_last_5","clearances_avg_last_5","tackles_avg_last_5","marks_avg_last_5"
    ]:
        if f"A_{name}" in Xy.columns and f"B_{name}" in Xy.columns:
            newcol = f"{name}_diff5"
            Xy[newcol] = Xy[f"A_{name}"] - Xy[f"B_{name}"]
            feature_cols.append(newcol)

    # Matchup vs opp concessions
    def add_matchup(a_key, b_concede_key, out):
        if f"A_{a_key}" in Xy.columns and f"B_{b_concede_key}" in Xy.columns:
            Xy[out] = Xy[f"A_{a_key}"] - Xy[f"B_{b_concede_key}"]
            feature_cols.append(out)

    add_matchup("disposals_avg_last_5", "concede_disposals_avg_last_5", "disp_matchup5")
    add_matchup("tackles_avg_last_5",   "concede_tackles_avg_last_5",   "tack_matchup5")
    add_matchup("marks_avg_last_5",     "concede_marks_avg_last_5",     "marks_matchup5")
    add_matchup("clearances_avg_last_5","concede_clearances_avg_last_5","clear_matchup5")

    # Rest days per team (as-of match), then rest_diff
    tg_days = (
        pd.read_sql('SELECT "Team","Date" FROM team_games', engine)
          .assign(Date=lambda d: pd.to_datetime(d["Date"], errors="coerce"))
          .dropna(subset=["Date"])
          .sort_values(["Team","Date"])
    )
    tg_days["days_since"] = tg_days.groupby("Team")["Date"].diff().dt.days
    Xy = (Xy.merge(tg_days.rename(columns={"Team":"team_a","days_since":"A_days_rest"}),
                   on=["Date","team_a"], how="left")
             .merge(tg_days.rename(columns={"Team":"team_b","days_since":"B_days_rest"}),
                   on=["Date","team_b"], how="left"))
    Xy["rest_diff"] = Xy["A_days_rest"] - Xy["B_days_rest"]
    feature_cols += ["rest_diff"]

    # NEW: interstate / secondary-home edges (A - B)
    if "A_is_interstate" in Xy.columns and "B_is_interstate" in Xy.columns:
        Xy["interstate_edge"] = (Xy["A_is_interstate"].fillna(0).astype(float)
                                 - Xy["B_is_interstate"].fillna(0).astype(float))
        feature_cols.append("interstate_edge")

    if "A_is_secondary_home" in Xy.columns and "B_is_secondary_home" in Xy.columns:
        Xy["secondary_home_edge"] = (Xy["A_is_secondary_home"].fillna(0).astype(float)
                                     - Xy["B_is_secondary_home"].fillna(0).astype(float))
        feature_cols.append("secondary_home_edge")

    # Fill NaNs in added numeric cols with column medians (robust)
    added = [c for c in feature_cols if c not in [
        "elo_diff","glicko_diff","rd_diff","vol_diff",
        "home_edge","ts_day","ts_night","ts_twilight","ts_unknown","elo_exp"
    ]]
    for c in added:
        Xy[c] = pd.to_numeric(Xy[c], errors="coerce")
        if Xy[c].isna().any():
            # travel edges default to 0 (no travel info) if median is NaN
            fill_val = 0.0 if c in ("interstate_edge","secondary_home_edge") else Xy[c].median()
            Xy[c] = Xy[c].fillna(fill_val)

    return Xy, feature_cols

Xy, feature_cols = add_team_precompute_features(Xy, feature_cols)

# ---------- Prepare matrices ----------
X = Xy[feature_cols].astype(float)
y = Xy["y"].astype(int)

# ---------- CV training + evaluation ----------
tscv = TimeSeriesSplit(n_splits=args.n_splits)
fold_metrics = []
oof_pred = np.zeros(len(X))
coef_records = []  # standardized LR coefs per fold
perm_imp_records = None

for fold, (tr, va) in enumerate(tscv.split(X), 1):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs"))
    ])
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ytr, yva = y.iloc[tr], y.iloc[va]

    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    oof_pred[va] = p

    m = {
        "fold": fold,
        "logloss":  float(log_loss(yva, p, labels=[0,1])),
        "brier":    float(brier_score_loss(yva, p)),
        "roc_auc":  float(roc_auc_score(yva, p)),
        "acc@0.5":  float(accuracy_score(yva, (p >= 0.5).astype(int))),
        "n_val":    int(len(yva)),
    }
    fold_metrics.append(m)

    lr = clf.named_steps["lr"]
    for fname, c in zip(feature_cols, lr.coef_[0]):
        coef_records.append({"fold": fold, "feature": fname, "coef": float(c), "abs_coef": float(abs(c))})

    if args.perm_importance and (fold == args.n_splits):
        r = permutation_importance(
            clf, Xva, yva, n_repeats=8, random_state=42, scoring="roc_auc"
        )
        perm_imp_records = pd.DataFrame({
            "feature": feature_cols,
            "perm_importance_mean": r.importances_mean,
            "perm_importance_std":  r.importances_std,
        })

# ---------- Train final model with CV-tuned C ----------
final_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegressionCV(
        Cs=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        cv=tscv,
        scoring="neg_log_loss",
        max_iter=4000,
        solver="lbfgs",
        refit=True,
    ))
])
final_clf.fit(X, y)

# ---------- Reliability (calibration) table on OOF ----------
bins = np.linspace(0.0, 1.0, 11)
cats = pd.cut(oof_pred, bins=bins, include_lowest=True)
rel = (pd.DataFrame({"bin": cats, "pred": oof_pred, "y": y})
         .groupby("bin")
         .agg(n=("y","size"), pred_mean=("pred","mean"), observed_rate=("y","mean"))
         .reset_index())
rel["bin_left"]  = rel["bin"].apply(lambda c: c.left)
rel["bin_right"] = rel["bin"].apply(lambda c: c.right)
rel = rel[["bin_left","bin_right","n","pred_mean","observed_rate"]]

# ---------- Aggregate feature importance ----------
coef_df = pd.DataFrame(coef_records)
coef_summary = (coef_df.groupby("feature", as_index=False)
                      .agg(coef_mean=("coef","mean"),
                           coef_std=("coef","std"),
                           abs_coef_mean=("abs_coef","mean"))
               .sort_values("abs_coef_mean", ascending=False))
if perm_imp_records is not None:
    feat_imp = coef_summary.merge(perm_imp_records, on="feature", how="left")
else:
    feat_imp = coef_summary
feat_imp["rank_coef"] = feat_imp["abs_coef_mean"].rank(method="min", ascending=False).astype(int)

# ---------- Persist artifacts ----------
joblib.dump(final_clf, MODEL_PATH)
META_PATH.write_text(json.dumps({
    "feature_cols": feature_cols,
    "cv": fold_metrics,
    "rows": int(len(X)),
    "from_table": "team_games",
    "used_team_precompute": bool("disp_matchup5" in feature_cols or any(c.endswith("_diff5") for c in feature_cols)),
    "exclude_finals": bool(args.exclude_finals),
    "n_splits": int(args.n_splits),
}, indent=2))

feat_imp.to_csv(IMP_CSV, index=False)
rel.to_csv(RELIAB_CSV, index=False)

# Also write to DB for easy querying
feat_imp.to_sql("team_winner_feature_importance", engine, if_exists="replace", index=False)
rel.to_sql("team_winner_reliability", engine, if_exists="replace", index=False)

# ---------- Console summary ----------
print("✅ Trained and saved:", MODEL_PATH)
print("📊 CV metrics:", json.dumps(fold_metrics, indent=2))
print("\n🏅 Top 15 features by |coef| (std-scaled):")
print(feat_imp[["feature","abs_coef_mean","coef_mean","coef_std"]].head(15).to_string(index=False))
print("\n📈 Reliability (saved):", RELIAB_CSV)
print("📑 Feature importance (saved):", IMP_CSV)
