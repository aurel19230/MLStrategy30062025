import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import normalized_mutual_info_score
from pynput import keyboard  # Importation pour la détection de touche
import os
from sklearn.metrics import normalized_mutual_info_score

from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement
file_name = "Step4_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort.csv"
# Chemin du répertoire
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version1"
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"



# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)

REPLACE_NAN = False
REPLACED_NANVALUE_BY = 90000.54789
REPLACED_NANVALUE_BY_INDEX = 1
if REPLACE_NAN:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")

# Configuration
CONFIG = {
    'FILE_PATH': file_path,
}

df = load_data(CONFIG['FILE_PATH'])

df_filtered = df[df['class_binaire'].isin([0, 1])].copy()

# Nouvelles features - Order Flow:
# Imbalances haussières
df_filtered['bull_imbalance_low_2'] = np.where(
    df_filtered['bidVolLow_1'] == 0,
    -6,
    np.where(
        (df_filtered['bidVolLow_1'] >= 1) & (df_filtered['bidVolLow_1'] <= 2),
        -2,
        df_filtered['askVolLow_2'] / df_filtered['bidVolLow_1']
    )
)

target_y = df_filtered['class_binaire'].copy()  # Série de la cible

# Fonction pour calculer l'information value (IV)
def calculate_iv(feature, target):
    df = pd.DataFrame({'feature': feature, 'target': target})
    cross_tab = pd.crosstab(df['feature'], df['target'])
    cross_tab = cross_tab + 0.5  # Éviter division par zéro

    print("\nCross-tabulation:")
    print(cross_tab)  # Debug

    # Vérifier si la classe 1 est présente
    if 1 not in cross_tab.columns or 0 not in cross_tab.columns:
        print("Warning: Missing class in cross-tab. Returning IV = 0.")
        return 0  # Retourne un IV nul si une classe est absente

    prop_event = cross_tab[1] / cross_tab[1].sum()
    prop_non_event = cross_tab[0] / cross_tab[0].sum()

    woe = np.log(prop_event / prop_non_event)
    iv = (prop_event - prop_non_event) * woe

    return iv.sum()

def calculate_power(feature, target):
    # Vérification si la feature contient bien les classes attendues
    unique_values = set(feature)
    if len(unique_values) < 2:
        print("Warning: Feature has insufficient unique values. Returning power = 0.")
        return 0  # Évite les erreurs dues à des variables constantes

    # Test du Chi2
    contingency = pd.crosstab(feature, target)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print("Warning: Contingency table does not contain enough classes. Returning power = 0.")
        return 0

    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    # Information Value (IV)
    iv = calculate_iv(feature, target)

    # Nouveau score combiné sans AUC
    # Exemple: 50% IV et 50% -log10(p_value)
    # Ajuste les pondérations à ta convenance
    score = 1 * iv + 0* (- np.log10(p_value + 1e-10))

    return score


def evaluate_feature(feature, target):
    # Calculer IV
    iv = calculate_iv(feature, target)

    # Calculer MI normalisée
    mi = normalized_mutual_info_score(feature, target)

    # Calculer la significativité statistique
    contingency = pd.crosstab(feature, target)
    chi2, p_value, _, _ = chi2_contingency(contingency)

    # Score combiné (exemple de pondération)
    score = (0.5 * iv) + (0.3 * mi) + (0.2 * (-np.log10(p_value + 1e-10)))

    return {
        'iv': iv,
        'mi': mi,
        'p_value': p_value,
        'combined_score': score
    }
# Choose which imbalance feature to optimize BEFORE running Optuna
# You can change this variable to select different features
SELECTED_FEATURE = 'bull_imbalance_low_2'  # Change this to any of your 12 features

# Map the selected feature to the appropriate volume columns
if SELECTED_FEATURE == 'bull_imbalance_low_1':
    bid_vol_col = 'bidVolLow'
    ask_vol_col = 'askVolLow_1'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_low_2':
    bid_vol_col = 'bidVolLow_1'
    ask_vol_col = 'askVolLow_2'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_low_3':
    bid_vol_col = 'bidVolLow_2'
    ask_vol_col = 'askVolLow_3'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_0':
    bid_vol_col = 'bidVolHigh_1'
    ask_vol_col = 'askVolHigh'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_1':
    bid_vol_col = 'bidVolHigh_2'
    ask_vol_col = 'askVolHigh_1'
    is_bull = True
elif SELECTED_FEATURE == 'bull_imbalance_high_2':
    bid_vol_col = 'bidVolHigh_3'
    ask_vol_col = 'askVolHigh_2'
    is_bull = True
# Bearish low features
elif SELECTED_FEATURE == 'bear_imbalance_low_0':
    ask_vol_col = 'askVolLow_1'
    bid_vol_col = 'bidVolLow'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_low_1':
    ask_vol_col = 'askVolLow_2'
    bid_vol_col = 'bidVolLow_1'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_low_2':
    ask_vol_col = 'askVolLow_3'
    bid_vol_col = 'bidVolLow_2'
    is_bull = False
# Adding bearish high features
elif SELECTED_FEATURE == 'bear_imbalance_high_1':
    ask_vol_col = 'askVolHigh'
    bid_vol_col = 'bidVolHigh_1'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_high_2':
    ask_vol_col = 'askVolHigh_1'
    bid_vol_col = 'bidVolHigh_2'
    is_bull = False
elif SELECTED_FEATURE == 'bear_imbalance_high_3':
    ask_vol_col = 'askVolHigh_2'
    bid_vol_col = 'bidVolHigh_3'
    is_bull = False
else:
    raise ValueError(f"Unknown feature: {SELECTED_FEATURE}")

# Print confirmation of which feature is being optimized
print(f"Optimizing feature: {SELECTED_FEATURE}")
if is_bull:
    print(f"Using bid volume column: {bid_vol_col}")
    print(f"Using ask volume column: {ask_vol_col}")
else:
    print(f"Using ask volume column: {ask_vol_col}")
    print(f"Using bid volume column: {bid_vol_col}")
print(f"Feature type: {'Bullish' if is_bull else 'Bearish'} imbalance")

from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
# Define your objective function using the selected columns
def objective(trial):
    # Paramètres pour les seuils de volume
    yy = trial.suggest_int('yy', 3, 3)
    z = trial.suggest_int('z', 18, 18)  # z doit être >= yy

    # Paramètres pour les seuils de bins
    c = trial.suggest_float('c', 0, 10)
    d = trial.suggest_float('d', c, 16)
    e = trial.suggest_float('e', d, 22)
    #f = trial.suggest_float('f', e, 16)

    # Constants for special values
    Imb_Div0 = -6
    Imb_zone = -2

    # Construction de la feature avec les seuils optimisés
    if is_bull:
        # For bullish features, we check if bid volume is too small and calculate ask/bid
        imbalance_raw = np.where(
            df_filtered[bid_vol_col] < yy,
            Imb_Div0,  # Valeur spéciale pour volume très faible
            np.where(
                (df_filtered[bid_vol_col] >= yy) & (df_filtered[bid_vol_col] <= z),
                Imb_zone,  # Valeur spéciale pour volume dans l'intervalle [yy, z]
                df_filtered[ask_vol_col] / df_filtered[bid_vol_col]  # Ratio standard pour les autres cas
            )
        )
    else:
        # For bearish features, we check if ask volume is too small and calculate bid/ask
        imbalance_raw = np.where(
            df_filtered[ask_vol_col] < yy,
            Imb_Div0,  # Valeur spéciale pour volume très faible
            np.where(
                (df_filtered[ask_vol_col] >= yy) & (df_filtered[ask_vol_col] <= z),
                Imb_zone,  # Valeur spéciale pour volume dans l'intervalle [yy, z]
                df_filtered[bid_vol_col] / df_filtered[ask_vol_col]  # Ratio standard pour les autres cas
            )
        )

    # Bins avec seuils dynamiques
    bins = [-np.inf, -5, 0, c, d, e,
            #f,
            np.inf]
    imbalance_binned = pd.cut(imbalance_raw, bins=bins, labels=False)

    # Rest of your evaluation code remains the same...


    # Vérifier le nombre de bins obtenus
    num_bins = pd.Series(imbalance_binned).nunique()
    print(f"Nombre de bins uniques après discrétisation: {num_bins}")

    # # Vérification et remplacement des NaN avant d'utiliser les valeurs
    # if imbalance_binned.isna().sum() > 0:
    #     print(f"Attention: {imbalance_binned.isna().sum()} valeurs NaN détectées après `pd.cut()`.")
    #     #imbalance_binned = np.where(pd.isna(imbalance_binned), -1, imbalance_binned)  # Remplace NaN par -1

    # Calcul de la puissance statistique
    power_score = calculate_power(imbalance_binned, target_y)

    # Vérification de la distribution (pénalise les bins trop vides)
    bin_counts = pd.Series(imbalance_binned).value_counts(normalize=True)
    min_bin_pct = bin_counts.min() if not bin_counts.empty else 0

    # Pénalité si un bin contient moins de 1% des données
    if min_bin_pct < 0.04:
        # penalty = (0.04 - min_bin_pct) * 10
        # power_score -= penalty
        return -np.inf

    mi_score = normalized_mutual_info_score(imbalance_binned, target_y)

    # **2) Test du Chi² pour voir si `imbalance_binned` est bien lié à `target_y`**
    contingency = pd.crosstab(imbalance_binned, target_y)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return -np.inf  # Si pas assez de diversité dans les bins

    chi2, p_value, _, _ = chi2_contingency(contingency)

    # **3) Score basé sur la significativité statistique**
    # - Si p-value faible → forte relation entre `imbalance_binned` et `target_y`
    # - On maximise `-log10(p-value)`, qui est grand si p-value est petite
    power_stat = -np.log10(p_value + 1e-10)

    # **4) Combinaison des scores**
    alpha = 1  # Poids pour l'information mutuelle
    beta = 0  # Poids pour la puissance statistique via Chi²

    score = alpha * mi_score + beta * power_stat

    ev = evaluate_feature(imbalance_binned, target_y)

    # {
    #     'iv': iv,
    #     'mi': mi,
    #     'p_value': p_value,
    #     'combined_score': score
    # }

    return ev['iv']


STOP_OPTIMIZATION = False

def callback_optuna_stop(study, trial):
    global STOP_OPTIMIZATION
    if STOP_OPTIMIZATION:
        print("Callback triggered: stopping the study.")
        study.stop()

def on_press(key):
    global STOP_OPTIMIZATION
    try:
        if key.char == '²':
            print("Stop signal received: stopping the study.")
            STOP_OPTIMIZATION = True
    except AttributeError:
        pass

# (Votre code existant jusqu'à l'appel à study.optimize...)

# Configurer et exécuter l'optimisation
study = optuna.create_study(direction='maximize')

# Démarrer le listener de clavier avant l'optimisation
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Lancer l'optimisation avec le callback d'arrêt
study.optimize(objective, n_trials=3000, callbacks=[callback_optuna_stop])

# Arrêter proprement le listener après l'optimisation
listener.stop()
listener.join()

# Meilleurs paramètres
best_params = study.best_params
print("Meilleurs paramètres trouvés:")
print(f"y = {best_params['yy']}, z = {best_params['z']}")  # Since yy is fixed at 1
print(f"Seuils de bins: c = {best_params['c']}, d = {best_params['d']}, e = {best_params['e']}"
      #f", f = {best_params['f']}"
      f"")

# Appliquer les meilleurs paramètres au dataset final
best_z =  best_params['z']
best_a, best_b = -5, 0
best_c, best_d, best_e= best_params['c'], best_params['d'], best_params['e'],

#best_f =best_params['f']

# Création de la variable optimisée
# When creating the optimized feature, use .loc for assignment
df_filtered.loc[:, 'bull_imbalance_low_2_optimized'] = np.where(
    df_filtered['bidVolLow_1'] == 0,
    -3.5,
    np.where(
        (df_filtered['bidVolLow_1'] >= 1) & (df_filtered['bidVolLow_1'] <= best_z),
        0,
        df_filtered['askVolLow_2'] / df_filtered['bidVolLow_1']
    )
)

# Discrétisation optimisée
best_bins = [-np.inf, best_a, best_b, best_c, best_d, best_e, np.inf]
df_filtered.loc[:, 'bull_imbalance_low_2_optimized'] = pd.cut(
    df_filtered['bull_imbalance_low_2_optimized'],
    bins=best_bins,
    labels=False
)


# Comparer les performances (originale vs optimisée)
original_power = calculate_power(df_filtered['bull_imbalance_low_2'], target_y)
optimized_power = calculate_power(df_filtered['bull_imbalance_low_2_optimized'], target_y)

print(f"\nPuissance statistique originale: {original_power:.4f}")
print(f"Puissance statistique optimisée: {optimized_power:.4f}")
print(f"Amélioration: {(optimized_power - original_power) / original_power * 100:.2f}%")

mi_score = normalized_mutual_info_score(df_filtered['bull_imbalance_low_2'], target_y)
mi_score_opti = normalized_mutual_info_score(df_filtered['bull_imbalance_low_2_optimized'], target_y)


print(f"\nInformation mutuelle normalisée originale: {mi_score:.6f}")
print(f"Information mutuelle normalisée optimisée: {mi_score_opti:.6f}")
print(f"Amélioration de l'information mutuelle: {((mi_score_opti - mi_score) / mi_score) * 100:.2f}%")

# Analyse détaillée des bins optimisés
bin_analysis = pd.crosstab(
    df_filtered['bull_imbalance_low_2_optimized'],
    target_y,
    normalize='index'
)
bin_counts = df_filtered['bull_imbalance_low_2_optimized'].value_counts().sort_index()

analysis_df = pd.DataFrame({
    'Bin': range(len(bin_counts)),
    'Count': bin_counts.values,
    'Percentage': bin_counts.values / len(df_filtered) * 100,
    'Target_Rate': bin_analysis[1]
})

print("\nAnalyse détaillée des bins optimisés:")
print(analysis_df)