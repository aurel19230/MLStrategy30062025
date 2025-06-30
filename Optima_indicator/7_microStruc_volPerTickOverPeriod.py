# -*- coding: utf-8 -*-
"""vol_per_tick_optimizer_numba.py – v4
============================================================
Ajout d'un **split TRAIN0** + contraintes MIN_TRADES sur tous
les splits utilisés dans le score.

• CSV_TRAIN0  +  flag  `USE_TRAIN0_IN_OPTIMIZATION`
• Si le flag est True, TRAIN0 est filtré + comptabilisé dans le score
• Chaque split actif doit respecter :
      – WR >= WINRATE_MIN
      – pct_trades >= PCT_TRADE_MIN
      – n_trades >= MIN_TRADES   (nouveau)
• TEST (split 5) reste un hold-out de contrôle, hors Optuna

Version avec fonction compute_volPerTickOverPeriod factorée
"""

from __future__ import annotations
from pathlib import Path
import sys, math, warnings, optuna, pandas as pd, numpy as np, chardet
from typing import Tuple
from Tools.func_features_preprocessing import prepare_arrays_for_optimization,compute_ratio_from_arrays

# ════════════════ Numba ════════════════════════════════════════════
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    sys.exit("❌  Numba n'est pas installé :  pip install numba")

warnings.filterwarnings("ignore")

# ─────────────── Paramètres globaux ───────────────────────────────
RANDOM_SEED = 42
DIRECTION = "Short"  # "Short" | "Long"

# activation par split
USE_TRAIN0_IN_OPTIMIZATION = False  # split 1 (TRAIN0)
USE_TRAIN_IN_OPTIMIZATION = True  # split 2 (TRAIN)

# hyper-paramètres
PERIOD_MIN, PERIOD_MAX = 15, 40
TL_MIN, TL_MAX, TL_STEP = 0.0, 2, 0.025
TH_MIN, TH_MAX = 2, 3.5

WINRATE_MIN, PCT_TRADE_MIN = 0.522, 0.055
MIN_TRADES = 10  # ▼ nouvelle contrainte

N_TRIALS, PRINT_EVERY = 50_000, 25
ALPHA, LAMBDA_WR, LAMBDA_PCT = 1, 1, 0  # ALPHA=1 : score = moyenne WR
FAILED_PENALTY = -1.0

# ════════════════ Chemins CSV ═════════════════════════════════════
DIR = (r"C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject"
       r"\\Sierra_chart\\xTickReversal\\simu\\5_0_5TP_6SL\\merge")
TEMPLATE = (DIR +
            rf"\\Step5_5_0_5TP_6SL_010124_200625_extractOnlyFullSession_Only{DIRECTION}_feat__split{{split}}.csv")

CSV_TRAIN0 = TEMPLATE.format(split="1_01012024_01052024")
CSV_TRAIN = TEMPLATE.format(split="2_01052024_30092024")
CSV_VAL = TEMPLATE.format(split="3_30092024_28022025")
CSV_TEST = TEMPLATE.format(split="4_02032025_15052025")
CSV_VAL1 = TEMPLATE.format(split="5_15052025_20062025")


# ════════════════ Chargement CSV + stats brutes ═══════════════════
def _enc(p: str, n=40_000):
    with open(p, "rb") as f:
        raw = f.read(n)
    enc = chardet.detect(raw)["encoding"]
    return "ISO-8859-1" if enc.lower() == "ascii" else enc


def load_csv(p):
    df = pd.read_csv(p, sep=";", encoding=_enc(p))
    df["sc_sessionStartEnd"] = pd.to_numeric(df["sc_sessionStartEnd"],
                                             errors="coerce").astype("Int16")
    df = df.dropna(subset=["sc_sessionStartEnd"])
    df["session_id"] = (df["sc_sessionStartEnd"] == 10).cumsum().astype("int32")
    df_f = df[df["class_binaire"].isin([0, 1])]
    return df, df_f, dict(
        wr=(df_f["class_binaire"] == 1).mean(),
        trades=len(df_f)
    )
T0_C, T0_F, ST_T0 = load_csv(CSV_TRAIN0)
TR_C, TR_F, ST_TR = load_csv(CSV_TRAIN)
V_C, V_F, ST_V = load_csv(CSV_VAL)
V1_C, V1_F, ST_V1 = load_csv(CSV_VAL1)
T_C, T_F, ST_T = load_csv(CSV_TEST)


# ════════════════ Pré-extraction ndarray (Numba ready) ════════════
# REMPLACÉ par la nouvelle fonction factorée
ARR = {
    "TR0": prepare_arrays_for_optimization(T0_C),
    "TR": prepare_arrays_for_optimization(TR_C),
    "V": prepare_arrays_for_optimization(V_C),
    "V1": prepare_arrays_for_optimization(V1_C),
    "T": prepare_arrays_for_optimization(T_C),
}


# ════════════════ Nouvelle fonction ratio ════════════════════════
def ratio(arr, period):
    """
    Calcule le ratio volume per tick à partir des arrays pré-calculés.

    Args:
        arr: Tuple (volume_per_tick, session_starts)
        period: Période pour la moyenne mobile

    Returns:
        Array des ratios normalisés
    """
    vpt, starts = arr
    return compute_ratio_from_arrays(vpt, starts, period)


# ════════════════ Kernels Numba (metrics seulement) ══════════════
@njit
def _metrics_nb(cls, sid, mask, base):
    wins = np.sum((cls == 1) & mask)
    loss = np.sum((cls == 0) & mask)
    tot = wins + loss
    if tot == 0:
        return 0., 0., 0, 0, 0
    wr = wins / tot
    pct = tot / base
    sess = 0
    prev = -1
    for i in range(mask.size):
        if mask[i] and sid[i] != prev:
            sess += 1
            prev = sid[i]
    return wr, pct, wins, loss, sess


# ════════════════ Signal helper ═══════════════════════════════════
def make_signal(df_f, arr, p, tl, th):
    r = ratio(arr, p)[df_f.index]
    r = np.where(np.isnan(r), 1.0, r)
    return (r <= tl) | (r >= th)  # hors bande


# ════════════════ Optuna objective ════════════════════════════════
best = {"score": -math.inf}


def objective(trial):
    global best
    p = trial.suggest_int("period", PERIOD_MIN, PERIOD_MAX)
    tl = trial.suggest_float("tl", TL_MIN, TL_MAX, step=TL_STEP)
    th = trial.suggest_float("th", TH_MIN, TH_MAX, step=TL_STEP)
    if tl >= th:
        return FAILED_PENALTY

    def m(df_f, key):
        mask = make_signal(df_f, ARR[key], p, tl, th)
        return _metrics_nb(
            df_f["class_binaire"].to_numpy(np.int8),
            df_f["session_id"].to_numpy(np.int32),
            mask,
            len(df_f),
        )

    wr_t0, pct_t0, suc_t0, fail_t0, _ = m(T0_F, "TR0")
    wr_t, pct_t, suc_t, fail_t, _ = m(TR_F, "TR")
    wr_v, pct_v, suc_v, fail_v, _ = m(V_F, "V")
    wr_v1, pct_v1, suc_v1, fail_v1, _ = m(V1_F, "V1")

    # ===== FILTRES MINIMA (WR / pct_trade) =====
    checks = [(wr_v, pct_v), (wr_v1, pct_v1)]
    if USE_TRAIN_IN_OPTIMIZATION:
        checks.append((wr_t, pct_t))
    if USE_TRAIN0_IN_OPTIMIZATION:
        checks.append((wr_t0, pct_t0))
    for wr_, pct_ in checks:
        if wr_ < WINRATE_MIN or pct_ < PCT_TRADE_MIN:
            return FAILED_PENALTY

    # ===== FILTRE MINIMUM   NOMBRE  DE   TRADES  =====
    trades_checks = [suc_v + fail_v, suc_v1 + fail_v1]
    if USE_TRAIN_IN_OPTIMIZATION:
        trades_checks.append(suc_t + fail_t)
    if USE_TRAIN0_IN_OPTIMIZATION:
        trades_checks.append(suc_t0 + fail_t0)
    for n_tr in trades_checks:
        if n_tr < MIN_TRADES:
            return FAILED_PENALTY

    # =====  SCORE (ALPHA=1 ⇒ moyenne WR)  =====
    wr_list = [wr_v, wr_v1]
    if USE_TRAIN_IN_OPTIMIZATION:
        wr_list.append(wr_t)
    if USE_TRAIN0_IN_OPTIMIZATION:
        wr_list.append(wr_t0)

    score = np.mean(wr_list)

    if score > best["score"]:
        best.update(
            number=trial.number,
            score=score,
            params=dict(period=p, tl=tl, th=th),
            wr_t0=wr_t0, pct_t0=pct_t0, trades_t0=suc_t0 + fail_t0,
            wr_t=wr_t, pct_t=pct_t, trades_t=suc_t + fail_t,
            wr_v=wr_v, pct_v=pct_v, trades_v=suc_v + fail_v,
            wr_v1=wr_v1, pct_v1=pct_v1, trades_v1=suc_v1 + fail_v1,
        )
    return score


# ════════════════ Boucle d'optimisation ═══════════════════════════
print("🚀  Optuna…")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
)
for it in range(1, N_TRIALS + 1):
    study.optimize(objective, n_trials=1)

    if it % PRINT_EVERY == 0 and best["score"] > -math.inf:
        p = best["params"]
        mask_t = make_signal(T_F, ARR["T"], p["period"], p["tl"], p["th"])
        wr_tst, pct_tst, suc_tst, fail_tst, _ = _metrics_nb(
            T_F["class_binaire"].to_numpy(np.int8),
            T_F["session_id"].to_numpy(np.int32),
            mask_t,
            len(T_F),
        )

        base_wrs = {
            "TR0": ST_T0["wr"],
            "TR": ST_TR["wr"],
            "V": ST_V["wr"],
            "V1": ST_V1["wr"],
            "T": ST_T["wr"],
        }


        def line(lbl, wr, pct, trades, base):
            print(f"   {lbl:<4} WR={wr:.2%}  pct={pct:.2%}  "
                  f"trades={trades}/{base}  baseWR={base_wrs[lbl]:.2%}")


        print(f"\n🟢 BEST so far (after {it} trials)")
        print(f"   params: VolPerTick-{DIRECTION} | period={p['period']} tl={p['tl']:.2f} th={p['th']:.2f}")

        if USE_TRAIN0_IN_OPTIMIZATION:
            line("TR0", best["wr_t0"], best["pct_t0"], best["trades_t0"], ST_T0["trades"])
        if USE_TRAIN_IN_OPTIMIZATION:
            line("TR", best["wr_t"], best["pct_t"], best["trades_t"], ST_TR["trades"])

        line("V", best["wr_v"], best["pct_v"], best["trades_v"], ST_V["trades"])
        line("V1", best["wr_v1"], best["pct_v1"], best["trades_v1"], ST_V1["trades"])
        line("T", wr_tst, pct_tst, suc_tst + fail_tst, ST_T["trades"])

# ════════════════ Récap final ═════════════════════════════════════
print("\n🏁  Optimisation terminée.")
if best["score"] > -math.inf:
    p = best["params"]
    mask_t = make_signal(T_F, ARR["T"], p["period"], p["tl"], p["th"])
    wr_tst, pct_tst, _, _, _ = _metrics_nb(
        T_F["class_binaire"].to_numpy(np.int8),
        T_F["session_id"].to_numpy(np.int32),
        mask_t,
        len(T_F),
    )

    print("────────  MEILLEUR ESSAI ────────")
    if USE_TRAIN0_IN_OPTIMIZATION:
        print(f"TR0  WR={best['wr_t0']:.2%}   pct={best['pct_t0']:.2%}")
    if USE_TRAIN_IN_OPTIMIZATION:
        print(f"TR   WR={best['wr_t']:.2%}   pct={best['pct_t']:.2%}")
    print(f"VAL  WR={best['wr_v']:.2%}   pct={best['pct_v']:.2%}")
    print(f"VAL1 WR={best['wr_v1']:.2%}   pct={best['pct_v1']:.2%}")
    print(f"TEST WR={wr_tst:.2%}   pct={pct_tst:.2%}")
else:
    print("❌  Aucun trial valide.")