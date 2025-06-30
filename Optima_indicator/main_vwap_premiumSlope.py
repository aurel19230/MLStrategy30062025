"""
Optuna + VWAP-Reversal-Pro (slopes fixes & CUSTOM) — full-candle optimisation
=============================================================================

•  Le signal est calculé sur *toutes* les bougies d’un split (df_full)
•  Les métriques WR / % trades sont mesurées sur df_lab (bougies label 0/1)
•  TRAIN / VAL / VAL1 / TEST ⇒ couple (df_full, df_lab)
•  Raccourcis : « & » test immédiat – « ² » stop propre + test final

Version “heavy-duty” : cache LRU, monitoring mémoire, multi-threads ,
pruner Médian, fast-math Numba.
"""

from __future__ import annotations

# ────────── Imports ────────────────────────────────────────────────
import os, time, psutil, concurrent.futures, optuna
from pathlib import Path
from typing import Tuple, Dict, Any
from functools import lru_cache

import numpy as np
import pandas as pd
from colorama import Fore, Style
from pynput import keyboard
from numba import jit, prange
from ta.volatility import AverageTrueRange
from optuna.pruners import MedianPruner

# ────────── Config ────────────────────────────────────────────────
DIR = Path(r"C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/"
           r"Sierra chart/xTickReversal/simu/5_0_5TP_6SL/merge")

CSV_TRAIN = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_" \
                  "OnlyShort_feat__split1_01012024_01052024.csv"
CSV_VAL   = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_" \
                  "OnlyShort_feat__split4_02032025_14052025.csv"
CSV_VAL1  = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_" \
                  "OnlyShort_feat__split3_01102024_28022025.csv"
CSV_TEST  = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_" \
                  "OnlyShort_feat__split2_01052024_01102024.csv"

WINRATE_MIN, PCT_TRADE_MIN = 0.54, 0.028
ALPHA, LAMBDA_WR, LAMBDA_PCT = 0.7, 0, 0
N_TRIALS, PRINT_EVERY = 10_000, 50
FAILED_PENALTY, RANDOM_SEED = -1e-3, 42

MAX_WORKERS        = max(1, os.cpu_count() - 1)      # multi-threads
MEMORY_LIMIT       = 0.80                            # 80 % de la RAM

# ────────── Runtime flags ─────────────────────────────────────────
STOP_OPTIMIZATION   = False   # touche « ² »
DF_TEST_CALCULATION = False   # touche « & »

def _on_press(key):
    global STOP_OPTIMIZATION, DF_TEST_CALCULATION
    if hasattr(key, "char"):
        if key.char == "²":
            print("\n🛑  Stop requested (²)")
            STOP_OPTIMIZATION = True
        elif key.char == "&":
            print("\n🧪  Test requested (&) – will run TEST after this trial")
            DF_TEST_CALCULATION = True

keyboard.Listener(on_press=_on_press, daemon=True).start()

# ────────── Data loading ──────────────────────────────────────────
def _read_csv(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    dtypes = {
        "close": "float32", "high": "float32", "low": "float32",
        "VWAP": "float32", "volume": "float32",
        "class_binaire": "int8", "SessionStartEnd": "float32",
    }
    df = pd.read_csv(path, sep=";", encoding="ISO-8859-1",
                     dtype=dtypes, parse_dates=["date"], low_memory=False)

    df["SessionStartEnd"] = df["SessionStartEnd"].fillna(0).astype("int16")
    df = df[df["SessionStartEnd"] > 0].reset_index(drop=True)

    n_start = (df["SessionStartEnd"] == 10).sum()
    n_end   = (df["SessionStartEnd"] == 20).sum()
    nb = min(n_start, n_end)
    print(f"{Fore.GREEN+'✔' if n_start==n_end else Fore.YELLOW+'⚠️'} {nb} sessions "
          f"dans {path.name}{Style.RESET_ALL}")

    df["session_id"] = (df["SessionStartEnd"] == 10).cumsum().astype("int32")
    #df_lab = df[df["class_binaire"].isin([0, 1])].copy().reset_index(drop=True)
    # on ne touche PAS à l’index : il doit rester celui de df
    df_lab = df[df["class_binaire"].isin([0, 1])].copy()
    return df.reset_index(drop=True), df_lab, nb


TR_FULL: pd.DataFrame; TR_LAB: pd.DataFrame
VA_FULL: pd.DataFrame; VA_LAB: pd.DataFrame
VA1_FULL: pd.DataFrame; VA1_LAB: pd.DataFrame
TE_FULL: pd.DataFrame; TE_LAB: pd.DataFrame

def load_all_datasets():
    global TR_FULL, TR_LAB, VA_FULL, VA_LAB, VA1_FULL, VA1_LAB, TE_FULL, TE_LAB
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_read_csv, p)
                for p in (CSV_TRAIN, CSV_VAL, CSV_VAL1, CSV_TEST)]
        TR_FULL, TR_LAB, _   = futs[0].result()
        VA_FULL, VA_LAB, _   = futs[1].result()
        VA1_FULL, VA1_LAB, _ = futs[2].result()
        TE_FULL, TE_LAB, _   = futs[3].result()

    for lbl, d in zip(("TR", "VA", "VA1", "TE"),
                      (TR_LAB, VA_LAB, VA1_LAB, TE_LAB)):
        print(f"{lbl:<4}| rows={len(d):,} | brute WR={(d['class_binaire']==1).mean():.2%}")
    print("────────────────────────────────────────────")

# ────────── Fast slope/std (Numba) ──────────────────────────────
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _slopes_numba(close_vals, session_starts, window):
    n = len(close_vals)
    slopes = np.full(n, np.nan, np.float32)
    r2s    = np.full(n, np.nan, np.float32)
    stds   = np.full(n, np.nan, np.float32)

    x = np.arange(1, window + 1, dtype=np.float32)
    sx, sxx, nf = x.sum(), (x * x).sum(), float(window)
    denom_base = nf * sxx - sx * sx

    for s in prange(n - window):
        e = s + window - 1
        r = s + window
        if r >= n or session_starts[s+1:e+1].any():
            continue

        y = close_vals[s:e+1]
        sy = y.sum()
        sxy = (x * y).sum()
        slope = (nf * sxy - sx * sy) / denom_base
        slope = max(min(slope, 1.), -1.)
        a = (sy - slope * sx) / nf

        diff = y - (a + slope * x)
        ssd  = (diff * diff).sum()
        std  = np.sqrt(ssd / nf) if window > 1 else 0.

        y_mean = sy / nf
        ss_tot = ((y - y_mean) ** 2).sum()
        r2     = 1. - (ssd / ss_tot) if ss_tot > 0 else 0.

        slopes[r] = slope
        r2s[r]    = r2
        stds[r]   = std
    return slopes, r2s, stds


@lru_cache(maxsize=256)
def _custom_slope_std(close_tup, start_tup, window, df_id):
    c = np.asarray(close_tup, np.float32)
    s = np.asarray(start_tup, np.bool_)
    slopes, _, stds = _slopes_numba(c, s, window)
    return slopes, stds

# ────────── VWAP-Reversal-Pro avec pentes ───────────────────────
def vwap_reversal_pro_slope(
    df: pd.DataFrame, *,
    lookback:int, momentum:int,
    z_window:int, atr_period:int, atr_mult:float,
    vol_lookback:int, vol_ratio_min:float,
    # pentes fixes
    use_slope_5:bool=True, use_slope_10:bool=True,
    use_slope_15:bool=True, use_slope_30:bool=True,
    slope_min_5:float=-0.8, slope_max_5:float=1,
    slope_min_10:float=-0.8, slope_max_10:float=1,
    slope_min_15:float=-0.8, slope_max_15:float=1,
    slope_min_30:float=-0.8, slope_max_30:float=1,
    std_min_5:float=0.5, std_max_5:float=3,
    std_min_10:float=0.5, std_max_10:float=3,
    std_min_15:float=0.5, std_max_15:float=3,
    std_min_30:float=0.5, std_max_30:float=3,
    # custom slope
    use_slope_custom:bool=False, custom_window:int=15,
    slope_min_c:float=-0.8, slope_max_c:float=1,
    std_min_c:float=0.5, std_max_c:float=3,
    slope_combination:str="MAJORITY",
) -> pd.Series:
    cols = [
        "session_id","close","VWAP","high","low","volume","SessionStartEnd",
        "sc_reg_slope_5P_2","sc_reg_std_5P_2",
        "sc_reg_slope_10P_2","sc_reg_std_10P_2",
        "sc_reg_slope_15P_2","sc_reg_std_15P_2",
        "sc_reg_slope_30P_2","sc_reg_std_30P_2",
    ]
    cols = [c for c in cols if c in df.columns]
    w = df[cols].copy()

    # custom slope/std
    if use_slope_custom:
        s_c, std_c = _custom_slope_std(
            tuple(df["close"].values),
            tuple((df["SessionStartEnd"] == 10).values),
            custom_window,
            id(df),
        )
        w["custom_slope"] = s_c
        w["custom_std"]   = std_c
    else:
        w["custom_slope"] = np.nan
        w["custom_std"]   = np.nan

    # indicateurs de base
    w["distance"]    = w["close"] - w["VWAP"]
    w["session_pos"] = w.groupby("session_id").cumcount() + 1

    w["z_dist"]  = np.nan
    w["speed"]   = np.nan
    w["mom"]     = np.nan
    w["atr"]     = np.nan
    w["dyn_th"]  = np.nan
    w["trend_ok"]= False
    w["vol_ok"]  = False

    for sid, idx in w.groupby("session_id").groups.items():
        session = w.loc[idx]

        # z-score
        if len(session) >= z_window:
            m  = session["distance"].rolling(z_window).mean()
            sd = session["distance"].rolling(z_window).std()
            v  = sd > 0
            z  = ((session["distance"] - m) / sd).replace([np.inf,-np.inf],0)
            w.loc[idx[v], "z_dist"] = z[v]

        # speed & momentum
        if len(session) > lookback:
            w.loc[idx, "speed"] = w.loc[idx, "z_dist"].diff(lookback)
        if len(session) > lookback + momentum:
            w.loc[idx, "mom"] = w.loc[idx, "speed"].diff(momentum)

        # ATR
        if len(session) >= atr_period:
            atr = AverageTrueRange(session["high"], session["low"],
                                   session["close"], atr_period,
                                   fillna=True).average_true_range()
            w.loc[idx, "atr"]     = atr.values
            w.loc[idx, "dyn_th"]  = atr.values * atr_mult

        # Trend flags (vectorisé)
        checks = []
        def _add(use, slope, std, smin, smax, dmin, dmax):
            if use:
                checks.append( slope.between(smin,smax) & std.between(dmin,dmax) )

        _add(use_slope_5 , session["sc_reg_slope_5P_2"] , session["sc_reg_std_5P_2"],
             slope_min_5 , slope_max_5 , std_min_5 , std_max_5)
        _add(use_slope_10, session["sc_reg_slope_10P_2"], session["sc_reg_std_10P_2"],
             slope_min_10, slope_max_10, std_min_10, std_max_10)
        _add(use_slope_15, session["sc_reg_slope_15P_2"], session["sc_reg_std_15P_2"],
             slope_min_15, slope_max_15, std_min_15, std_max_15)
        _add(use_slope_30, session["sc_reg_slope_30P_2"], session["sc_reg_std_30P_2"],
             slope_min_30, slope_max_30, std_min_30, std_max_30)
        _add(use_slope_custom, session["custom_slope"], session["custom_std"],
             slope_min_c, slope_max_c, std_min_c, std_max_c)

        if checks:
            bool_mat = np.vstack([c.values for c in checks])
            if slope_combination == "ALL":
                flags = bool_mat.all(axis=0)
            elif slope_combination == "ANY":
                flags = bool_mat.any(axis=0)
            else:  # MAJORITY
                flags = bool_mat.sum(axis=0) > (bool_mat.shape[0] / 2)
            w.loc[idx, "trend_ok"] = flags

        # volume
        if "volume" in session and len(session) >= vol_lookback:
            ma = session["volume"].rolling(vol_lookback).mean()
            vr = session["volume"] / ma
            w.loc[idx, "vol_ok"] = (vr > vol_ratio_min).fillna(False)

    # ---- Signal final
    signal = (
        (w["z_dist"] > 0) &
        (w["speed"]  > 0) &
        ((w["mom"] < -w["dyn_th"]) | w["trend_ok"]) &
        w["vol_ok"] &
        w["atr"].notna()
    )

    return signal.fillna(False)

# ────────── Cache & perf ─────────────────────────────────────────
_SIGNAL_CACHE: Dict[Tuple[int, Tuple[Tuple[str, Any], ...]], pd.Series] = {}
_DATASET_REGISTRY: Dict[int, pd.DataFrame] = {}

def _metrics(df_lab: pd.DataFrame, sig_lab: pd.Series):
    sub = df_lab.loc[sig_lab]  # indexation ALIGNÉE sur l’index
    return ((sub["class_binaire"] == 1).mean() if not sub.empty else 0.0,
            len(sub) / len(df_lab))

@lru_cache(maxsize=256)
def _cached_signal(df_id: int, params_tuple: Tuple[Tuple[str, Any], ...]):
    key = (df_id, params_tuple)
    if key in _SIGNAL_CACHE:
        return _SIGNAL_CACHE[key]
    df = _DATASET_REGISTRY[df_id]
    sig = vwap_reversal_pro_slope(df, **dict(params_tuple))
    if len(_SIGNAL_CACHE) > 1000:
        for k in list(_SIGNAL_CACHE.keys())[:200]:
            del _SIGNAL_CACHE[k]
    _SIGNAL_CACHE[key] = sig
    return sig

def perf(df_full, df_lab, params):
    df_id = id(df_full)
    _DATASET_REGISTRY[df_id] = df_full
    sig_full = _cached_signal(df_id, tuple(sorted(params.items())))
    return _metrics(df_lab, sig_full.loc[df_lab.index])

# ────────── Memory monitoring ───────────────────────────────────
def monitor_memory():
    mem = psutil.Process(os.getpid()).memory_info().rss / psutil.virtual_memory().total
    if mem > MEMORY_LIMIT:
        _SIGNAL_CACHE.clear()
        print(f"\n⚠️  RAM>{MEMORY_LIMIT*100:.0f}% ({mem*100:.1f}%) — cache vidé")
    return mem

# ────────── Test evaluation helper ─────────────────────────────
def evaluate_test(params: Dict[str, Any], final: bool = False):
    print("\n🧮  Calcul final sur TEST" if final else "\n🧪  TEST en cours…")
    wr, pct = perf(TE_FULL, TE_LAB, params)
    n = len(TE_LAB); ntr = int(pct * n)
    print(f"Trades {ntr:,}/{n:,} ({pct:.2%}) WR={wr:.2%}")
    if final:
        print("✅ VALIDE" if (wr >= WINRATE_MIN and pct >= PCT_TRADE_MIN) else "❌ REJET")

# ────────── Optuna objective ───────────────────────────────────
_best: Dict[str, Any] = {"score": -1}

def _define_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    p = dict(
        lookback      = trial.suggest_int("lookback", 15, 50),
        momentum      = trial.suggest_int("momentum", 3, 14),
        z_window      = trial.suggest_int("z_window", 15, 55),
        atr_period    = trial.suggest_int("atr_period", 12, 55),
        atr_mult      = round(trial.suggest_float("atr_mult", 0.5, 5)),
        vol_lookback  = trial.suggest_int("vol_lookback", 2, 15),
        vol_ratio_min = round(trial.suggest_float("vol_ratio_min", 0, 2), 2),
    )
    for w in (5, 10, 15, 30):
        p[f"use_slope_{w}"] = trial.suggest_categorical(f"use_slope_{w}", [True, False])
        p[f"slope_min_{w}"] = trial.suggest_float(f"slope_min_{w}", -1, 0.2)
        p[f"slope_max_{w}"] = trial.suggest_float(f"slope_max_{w}", 0, 1)
        p[f"std_min_{w}"]   = trial.suggest_float(f"std_min_{w}", 0, 0.6)
        p[f"std_max_{w}"]   = trial.suggest_float(f"std_max_{w}", 0.6, 3)
    p.update(
        use_slope_custom = trial.suggest_categorical("use_slope_custom", [True, True]),
        custom_window    = trial.suggest_int("custom_window", 10, 150),
        slope_min_c      = trial.suggest_float("slope_min_c", -1, 0.2),
        slope_max_c      = trial.suggest_float("slope_max_c", 0, 1),
        std_min_c        = trial.suggest_float("std_min_c", 0, 0.6),
        std_max_c        = trial.suggest_float("std_max_c", 0.6, 3),
        slope_combination= trial.suggest_categorical("slope_combination",
                                                     ["ALL", "ANY", "MAJORITY"]),
    )
    return p

def objective(trial: optuna.trial.Trial) -> float:
    global _best, DF_TEST_CALCULATION
    params = _define_params(trial)

    # checks
    for w in (5, 10, 15, 30):
        if params[f"slope_min_{w}"] >= params[f"slope_max_{w}"] or \
           params[f"std_min_{w}"]   >= params[f"std_max_{w}"]:
            return FAILED_PENALTY
    if params["slope_min_c"] >= params["slope_max_c"] or \
       params["std_min_c"]   >= params["std_max_c"]:
        return FAILED_PENALTY
    if not (any(params[f"use_slope_{w}"] for w in (5,10,15,30)) or
            params["use_slope_custom"]):
        return FAILED_PENALTY

    wr_t, pct_t = perf(TR_FULL,  TR_LAB,  params)
    wr_v, pct_v = perf(VA_FULL,  VA_LAB,  params)
    wr_v1,pct_v1= perf(VA1_FULL, VA1_LAB, params)

    if any([wr_t<WINRATE_MIN,pct_t<PCT_TRADE_MIN,
            wr_v<WINRATE_MIN,pct_v<PCT_TRADE_MIN,
            wr_v1<WINRATE_MIN,pct_v1<PCT_TRADE_MIN]):
        return FAILED_PENALTY

    gap_wr  = max(abs(wr_t-wr_v),  abs(wr_t-wr_v1),  abs(wr_v-wr_v1))
    gap_pct = max(abs(pct_t-pct_v),abs(pct_t-pct_v1),abs(pct_v-pct_v1))
    score   = ALPHA*((wr_t+wr_v+wr_v1)/3) + (1-ALPHA)*((pct_t+pct_v+pct_v1)/3) \
              - LAMBDA_WR*gap_wr - LAMBDA_PCT*gap_pct

    print(f"{trial.number:5d}| TR {wr_t:5.2%}/{pct_t:5.2%} | "
          f"VA {wr_v:5.2%}/{pct_v:5.2%} | "
          f"VA1 {wr_v1:5.2%}/{pct_v1:5.2%} → {score:.4f}")

    if score > _best["score"]:
        _best = dict(score=score, number=trial.number,
                     wr_t=wr_t, pct_t=pct_t, wr_v=wr_v, pct_v=pct_v,
                     wr_v1=wr_v1, pct_v1=pct_v1, params=params)

    if DF_TEST_CALCULATION and score > 0.5:
        evaluate_test(params)
    return score

# ────────── Optuna loop ─────────────────────────────────────────
def run_optimization():
    global DF_TEST_CALCULATION, STOP_OPTIMIZATION   # ← ajout
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = MedianPruner(n_startup_trials=5,
                          n_warmup_steps=10,
                          interval_steps=1)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED,
                                           multivariate=True),
        pruner=pruner)

    t0 = time.time()

    for i in range(N_TRIALS):
        if STOP_OPTIMIZATION:
            break

        study.optimize(objective, n_trials=1, show_progress_bar=False)

        if (i + 1) % 10 == 0:
            monitor_memory()

        if (i + 1) % PRINT_EVERY == 0 and _best["score"] > -1:
            elapsed = time.time() - t0
            tph = (i + 1) / (elapsed / 3600)
            bt = _best
            print(f"\n=== Trial {i+1}/{N_TRIALS} ({(i+1)/N_TRIALS:.1%}) | "
                  f"{elapsed/60:.1f} min | {tph:.1f} trials/h ===")

            for lbl, wr, pct, tot in (
                ("TR",  bt["wr_t"],  bt["pct_t"],  len(TR_LAB)),
                ("VA",  bt["wr_v"],  bt["pct_v"],  len(VA_LAB)),
                ("VA1", bt["wr_v1"], bt["pct_v1"], len(VA1_LAB))
            ):
                print(f"{lbl}: WR {wr:.2%} | {pct:.2%} ({int(pct * tot):,}/{tot:,})")

            print(f"Best score: {bt['score']:.4f} (trial #{bt['number']})")
            print("Paramètres principaux :")
            for k, v in bt['params'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            # Lancer systématiquement une évaluation sur TEST
            evaluate_test(bt["params"])


    print("\n🔚 Optuna finished")
    evaluate_test(_best["params"], final=True)
    return study



# ────────── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("💻  VWAP-Premium optimisation — démarrage")
    load_all_datasets()
    run_optimization()
