import platform, os
import pandas as pd
import numpy as np

# ───────────────────────── 1. Chargement CSV ─────────────────────────
file_name = "Step4__010124_120525_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
if platform.system() != "Darwin":
    directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL_noDOM_Janv2024_9Mai2025\merge"
else:
    directory_path = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"
csv_path = os.path.join(directory_path, file_name)
df = pd.read_csv(csv_path, encoding='iso-8859-1', sep=';')
features_df = df.copy()

# ───────────────────────── 2. Construction du signal tendance ─────────────────────────
def compute_consecutive_trend_feature(df, features_df, target_col, n=2, trend_type='up', output_col='trend_feature'):
    cond = pd.Series(True, index=df.index)
    for i in range(n):
        if trend_type == 'up':
            cond &= df[target_col].shift(i) >= df[target_col].shift(i + 1)
        elif trend_type == 'down':
            cond &= df[target_col].shift(i) <= df[target_col].shift(i + 1)
        else:
            raise ValueError("trend_type must be 'up' or 'down'")
        cond &= ~df[target_col].shift(i).isna()
        cond &= ~df[target_col].shift(i + 1).isna()
    features_df[output_col] = cond.astype(int)

# ───────────────────────── 3. Métriques d’un signal ─────────────────────────
def eval_trend_signal(features_df, trend_col, class_col='class_binaire'):
    nb_samples = (features_df[trend_col] == 1).sum()
    mask = (features_df[trend_col] == 1) & (features_df[class_col].isin([0, 1]))
    nb_trades = mask.sum()
    nb_wins = (features_df.loc[mask, class_col] == 1).sum()
    winrate = nb_wins / nb_trades if nb_trades else np.nan
    return nb_samples, nb_trades, nb_wins, winrate

# ───────────────────────── 4. Benchmark avec filtre R² ─────────────────────────
def benchmark_trends_with_r2(df, features_df, target_cols, n_values=(3, 5, 10),
                             trend_type='up', r2_threshold=0.75, class_col='class_binaire'):

    results = []

    for col in target_cols:
        for n in n_values:
            base_sig = f"{col}_{trend_type}_{n}"

            # 4-A. Signal brut
            compute_consecutive_trend_feature(df, features_df, col, n, trend_type, base_sig)
            nb_trades_brut = features_df[class_col].isin([0, 1]).sum()

            ns, nt, nw, wr = eval_trend_signal(features_df, base_sig, class_col)
            results.append({
                'signal': base_sig, 'type': 'raw', 'source_col': col, 'n': n,
                'nb_samples': ns, 'nb_trades': nt, 'nb_wins': nw, 'winrate': wr,
                'nb_trades_brut': nb_trades_brut
            })

            # 4-B. Signal filtré par R² (si col est un slope)
            if "vix_slope_" in col:
                r2_col = col.replace("slope", "r2")
                if r2_col in df.columns:
                    filt_name = f"{base_sig}_r2>{r2_threshold:.2f}"
                    mask_confirm = (features_df[base_sig] == 1) & (df[r2_col] > r2_threshold)
                    features_df[filt_name] = 0
                    features_df.loc[mask_confirm, filt_name] = 1

                    ns2, nt2, nw2, wr2 = eval_trend_signal(features_df, filt_name, class_col)
                    results.append({
                        'signal': filt_name, 'type': 'r2filtered', 'source_col': col, 'n': n,
                        'nb_samples': ns2, 'nb_trades': nt2, 'nb_wins': nw2, 'winrate': wr2,
                        'nb_trades_brut': nb_trades_brut
                    })
                    #features_df.drop(columns=[filt_name], inplace=True)

            # Nettoyage du signal brut
            #features_df.drop(columns=[base_sig], inplace=True)

    return pd.DataFrame(results)

# Colonnes à tester
target_cols = ['vix_slope_6', 'vix_slope_12', 'vix_slope_24',
               'vix_r2_6', 'vix_r2_12', 'vix_r2_24']

# ───────────────────────── 5. Lancement ─────────────────────────
benchmark_df = benchmark_trends_with_r2(df, features_df,
                                        target_cols=target_cols,
                                        n_values=(2,3, 4,5,8, 10,11,12,13,14,15,16,17,18,20,23),
                                        trend_type='up',
                                        r2_threshold=0.02,
                                        class_col='class_binaire')

# ───────────────────────── 6. Affichage top signaux ─────────────────────────
best = (benchmark_df
        .query("nb_trades >= 100")
        .sort_values(['winrate', 'nb_trades'], ascending=[False, False])
        .head(40))

print("\n🟢 Top 10 signaux (raw & r2filtered, nb_trades ≥100)")
print(best.to_string(index=False, formatters={
    'winrate': '{:.2%}'.format,
    'nb_trades_brut': '{:,.0f}'.format
}))
# Exemple de vote simple à 3 filtres
features_df['vote'] = (
    (features_df['vix_slope_12_up_2'] == 1).astype(int) +
    (features_df['vix_slope_24_up_23'] == 1).astype(int) +
    (features_df['vix_slope_12_up_2'] == 1).astype(int)
)
features_df['signal_final'] = (features_df['vote'] >= 2).astype(int)


# Choisir un seuil : au moins 2/3 accords
features_df['signal_final'] = (features_df['vote'] >= 2).astype(int)
# ─────────────────────────────────────────────────────
#  Évaluation de signal_final
# ─────────────────────────────────────────────────────
mask_signal = (features_df['signal_final'] == 1)

# (1) Nombre total de bougies où signal_final == 1
nb_samples_final = mask_signal.sum()

# (2) Masque « trades valides » : signal_final == 1 ET class_binaire ∈ {0,1}
mask_trades = mask_signal & (features_df['class_binaire'].isin([0, 1]))
nb_trades_final = mask_trades.sum()

# (3) Gagnants dans ces trades
nb_wins_final = (features_df.loc[mask_trades, 'class_binaire'] == 1).sum()

# (4) Win-rate
winrate_final = nb_wins_final / nb_trades_final if nb_trades_final else np.nan

# Affichage
print("🔎  Résultats pour signal_final")
print(f"- Bougies où signal_final == 1 : {nb_samples_final}")
print(f"- Trades pris (class 0/1)     : {nb_trades_final}")
print(f"- Trades gagnants (class 1)   : {nb_wins_final}")
print(f"- Win-rate                    : {winrate_final:.2%}")
