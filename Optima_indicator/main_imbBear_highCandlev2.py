# optuna_bear_imbalance_flexible.py – v7.2
# ---------------------------------------------------------
# ➊ Choix interactif : light / agressif / light+poc=4
# ➋ Filtrage WR·min / pct·min sur TRAIN, VAL, VAL1
# ➌ 32 threads Optuna, masques NumPy+Numba
# ➍ Évaluation finale sur TEST + export CSV
# ➎ Touches '&' pour tester sur TEST et '*' pour arrêter/reprendre
# ---------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np, optuna
from numba import njit
import chardet, sys, threading
from pynput import keyboard

# ──────────────────────────────────────────────────────────
# 1. CONFIGS PRÉ-DÉFINIES
# ──────────────────────────────────────────────────────────
CONFIGS = {
    "light": {
        "WINRATE_MIN": 0.52,
        "PCT_TRADE_MIN": 0.01,
        "ALPHA": 0.70,
    },
    "aggressive": {
        "WINRATE_MIN": 0.53,
        "PCT_TRADE_MIN": 0.015,
        "ALPHA": 0.75,
    },
}

ASK_MIN_1, ASK_MAX_1 = 1, 10
BEAR_MIN_1, BEAR_MAX_1 = 2, 20
ASK_MIN_2, ASK_MAX_2 = 10, 40
BEAR_MIN_2, BEAR_MAX_2 = 2, 20
ASK_MIN_3, ASK_MAX_3 = 40, 200
BEAR_MIN_3, BEAR_MAX_3 = 1.5, 20

# Optuna
RANDOM_SEED = 42
N_TRIALS = 10_000
PRINT_EVERY = 50
FAILED_PENALTY = -0.001

# Variables globales pour le contrôle
RUN_TEST = False
PAUSE_OPTIMIZATION = False

# ──────────────────────────────────────────────────────────
# 2. CHOIX UTILISATEUR
# ──────────────────────────────────────────────────────────
# D'abord, définir la variable seuil_poc
seuil_poc = -0.25  # ou une autre valeur selon votre besoin

# Ensuite, utiliser cette variable dans votre f-string
choice = input(
    f"Filtrage :\n"
    f"  [Entrée] → light\n"
    f"  a        → agressif\n"
    f"  z        → light + poc == {seuil_poc}\n"
    f"Choix : "
).strip().lower()
if choice == "a":
    cfg = CONFIGS["aggressive"]
    FILTER_POC = False
elif choice == "z":
    cfg = CONFIGS["light"]
    FILTER_POC = True
else:
    cfg = CONFIGS["light"]
    FILTER_POC = False

print(f"\n→ Mode : {'agressif' if choice == 'a' else 'light'}"
      f"{' + poc=-0.75' if FILTER_POC else ''}\n")

WINRATE_MIN = cfg["WINRATE_MIN"]
PCT_TRADE_MIN = cfg["PCT_TRADE_MIN"]
ALPHA = cfg["ALPHA"]

# ──────────────────────────────────────────────────────────
# 3. CHEMINS CSV
# ──────────────────────────────────────────────────────────
DIR = Path(r"C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/"
           r"Sierra chart/xTickReversal/simu/5_0_5TP_6SL/merge")

CSV_TRAIN = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split1_01012024_01052024.csv"
CSV_TEST = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split2_01052024_01102024.csv"
CSV_VAL1 = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
CSV_VAL = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"


# ──────────────────────────────────────────────────────────
# 4. CHARGEMENT & CONVERSION → NumPy
# ──────────────────────────────────────────────────────────
def load_to_numpy(path: Path, filter_poc: bool):
    df = pd.read_csv(path, sep=";", encoding="ISO-8859-1", parse_dates=["date"], low_memory=False)

    print(f"{path.name}")

    # ① on garde d'abord les cibles valides
    df = df[df["class_binaire"].isin([0, 1])].copy()

    # ② puis, si demandé, on filtre poc == 4
    if filter_poc and "poc" in df.columns:
        df = df[df["diffPriceClosePoc_0_0"] >= seuil_poc]

    df["session_id"] = (df["SessionStartEnd"] == 10).cumsum().astype("int32")

    return {
        "ask": df["askVolHigh"].to_numpy("float32"),
        "bear": df["bear_imbalance_high_1"].to_numpy("float32"),
        "y": df["class_binaire"].to_numpy("int8"),
        "sid": df["session_id"].to_numpy("int32"),
        "n_sess": int(df["session_id"].max()) + 1,
    }


DATA = {
    "tr": load_to_numpy(CSV_TRAIN, FILTER_POC),
    "v": load_to_numpy(CSV_VAL, FILTER_POC),
    "v1": load_to_numpy(CSV_VAL1, FILTER_POC),
    "test": load_to_numpy(CSV_TEST, FILTER_POC),
}


# ──────────────────────────────────────────────────────────
# 5. NUMBA : métriques
# ──────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _metrics(mask, y, sid, n_sess):
    tot = mask.sum()
    if tot == 0:
        return 0., 0.
    wr = (y & mask).sum() / tot
    pct = tot / y.size
    return wr, pct


# ──────────────────────────────────────────────────────────
# 6. OBJECTIVE OPTUNA
# ──────────────────────────────────────────────────────────
def objective(trial: optuna.trial.Trial) -> float:
    # Vérifier si l'optimisation est en pause
    global PAUSE_OPTIMIZATION
    if PAUSE_OPTIMIZATION:
        print("Optimisation en pause. Appuyez sur '*' pour reprendre.")
        while PAUSE_OPTIMIZATION:
            # Attente active avec petite pause pour réduire l'utilisation CPU
            import time
            time.sleep(0.1)
        print("Optimisation reprise!")

    p = {
        "ask1": trial.suggest_int("askVolHigh", ASK_MIN_1, ASK_MAX_1),
        "bear1": trial.suggest_float("bear_imbalance_high_1", BEAR_MIN_1, BEAR_MAX_1),
        "ask2": trial.suggest_int("askVolHigh_2Cond", ASK_MIN_2, ASK_MAX_2),
        "bear2": trial.suggest_float("bear_imbalance_high_1_2Cond", BEAR_MIN_2, BEAR_MAX_2),
        "ask3": trial.suggest_int("askVolHigh_3Cond", ASK_MIN_3, ASK_MAX_3),
        "bear3": trial.suggest_float("bear_imbalance_high_1_3Cond", BEAR_MIN_3, BEAR_MAX_3),
    }

    def dataset_metrics(d):
        m1 = (d["ask"] > p["ask1"]) & (d["bear"] > p["bear1"])
        m2 = (d["ask"] > p["ask2"]) & (d["bear"] > p["bear2"])
        m3 = (d["ask"] > p["ask3"]) & (d["bear"] > p["bear3"])
        return _metrics(m1 | m2 | m3, d["y"], d["sid"], d["n_sess"])

    wr_t, pct_t = dataset_metrics(DATA["tr"])
    wr_v, pct_v = dataset_metrics(DATA["v"])
    wr_v1, pct_v1 = dataset_metrics(DATA["v1"])

    if (wr_t < WINRATE_MIN or pct_t < PCT_TRADE_MIN or
            wr_v < WINRATE_MIN or pct_v < PCT_TRADE_MIN or
            wr_v1 < WINRATE_MIN or pct_v1 < PCT_TRADE_MIN):
        return FAILED_PENALTY

    score = ALPHA * (wr_t + wr_v + wr_v1) / 3 + (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3

    trial.set_user_attr("wr_tr", wr_t);
    trial.set_user_attr("pct_tr", pct_t)
    trial.set_user_attr("wr_val", wr_v);
    trial.set_user_attr("pct_val", pct_v)
    trial.set_user_attr("wr_val1", wr_v1);
    trial.set_user_attr("pct_val1", pct_v1)

    return score


# ──────────────────────────────────────────────────────────
# 7. TEST FINAL
# ──────────────────────────────────────────────────────────
def evaluate_on_test(params: dict):
    print("\n----- ÉVALUATION SUR TEST -----")
    d = DATA["test"]
    m1 = (d["ask"] > params["askVolHigh"]) & (d["bear"] > params["bear_imbalance_high_1"])
    m2 = (d["ask"] > params["askVolHigh_2Cond"]) & (d["bear"] > params["bear_imbalance_high_1_2Cond"])
    m3 = (d["ask"] > params["askVolHigh_3Cond"]) & (d["bear"] > params["bear_imbalance_high_1_3Cond"])

    # Masque combiné (union)
    m_combined = m1 | m2 | m3

    # Statistiques générales
    wr, pct = _metrics(m_combined, d["y"], d["sid"], d["n_sess"])

    # Calcul détaillé pour les affichages
    total_trades = m_combined.sum()
    success_trades = (d["y"] & m_combined).sum()
    failed_trades = total_trades - success_trades

    print(f"📊 TEST final → WR {wr:.2%} | pct {pct:.2%}")
    print(f"Trades: ✓{success_trades} ✗{failed_trades} Total={total_trades}")
    print("---------------------------")

    return wr, pct


# ──────────────────────────────────────────────────────────
# 8. KEYBOARD LISTENER
# ──────────────────────────────────────────────────────────
def on_press(key):
    global RUN_TEST, PAUSE_OPTIMIZATION
    try:
        if key.char == '&':
            print("\n🧪 Test demandé via '&' - Utilisation des meilleurs paramètres trouvés")
            RUN_TEST = True
        elif key.char == '*':
            PAUSE_OPTIMIZATION = not PAUSE_OPTIMIZATION
            status = "PAUSE" if PAUSE_OPTIMIZATION else "REPRISE"
            print(f"\n⏯️ {status} de l'optimisation via '*'")
    except AttributeError:
        # Touche spéciale sans caractère
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    print("⌨️ Écouteur clavier démarré:")
    print("   - '&' pour tester sur le dataset TEST")
    print("   - '*' pour arrêter/reprendre l'optimisation")
    return listener


# ──────────────────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────────────────
def main() -> None:
    # Démarrer le listener clavier
    listener = start_keyboard_listener()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

    for k in range(1, N_TRIALS + 1):
        # Vérifier si un test est demandé
        global RUN_TEST
        if RUN_TEST:
            RUN_TEST = False
            print("\n🔍 Test avec les MEILLEURS PARAMÈTRES trouvés jusqu'à présent:")
            print(f"Paramètres: {study.best_params}")
            evaluate_on_test(study.best_params)

        study.optimize(objective, n_trials=1, n_jobs=32, catch=(Exception,))

        if k % PRINT_EVERY == 0:
            bt = study.best_trial
            wr_tr = bt.user_attrs.get("wr_tr", np.nan)
            pct_tr = bt.user_attrs.get("pct_tr", np.nan)
            wr_v = bt.user_attrs.get("wr_val", np.nan)
            pct_v = bt.user_attrs.get("pct_val", np.nan)
            wr_v1 = bt.user_attrs.get("wr_val1", np.nan)
            pct_v1 = bt.user_attrs.get("pct_val1", np.nan)
            print(
                f"after {k:>6} trials – best score {bt.value:.4f} | "
                f"TR  WR {wr_tr:.2%} pct {pct_tr:.2%} | "
                f"V1  WR {wr_v:.2%} pct {pct_v:.2%} | "
                f"V2  WR {wr_v1:.2%} pct {pct_v1:.2%}"
            )

    best_params = study.best_params
    wr_test, pct_test = evaluate_on_test(best_params)

    print("\n── FIN OPTUNA ──")
    print("meilleurs paramètres :", best_params)
    print("meilleur score       :", study.best_value)
    print(f"📊 TEST final → WR {wr_test:.2%} | pct {pct_test:.2%}")

    study.trials_dataframe(attrs=("number", "value", "user_attrs")) \
        .to_csv("optuna_log_winrate_pct.csv", index=False)
    print("📁 Log CSV ➜ optuna_log_winrate_pct.csv")


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()