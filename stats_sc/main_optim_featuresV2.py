import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score

import os

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


from sklearn.metrics import normalized_mutual_info_score
def evaluate_feature(feature, target):
    # Calculer IV
    iv = calculate_iv(feature, target)

    # Calculer MI normalisée
    mi = normalized_mutual_info_score(feature, target)

    # Calculer la significativité statistique
    contingency = pd.crosstab(feature, target)
    chi2, p_value, _, _ = chi2_contingency(contingency)

    # Score combiné (exemple de pondération)
    score = (0.65 * iv) + (0.35 * mi) + (0 * (-np.log10(p_value + 1e-10)))

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
# Constants for special values
Imb_Div0 = -6
Imb_zone = -2
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
# Define your objective function using the selected columns
def objective(trial):
    # Paramètres pour les seuils de volume
    yy = trial.suggest_int('yy', 3, 3)
    z = trial.suggest_int('z', 18, 18)  # z doit être >= yy

    # Paramètres pour les seuils de bins
    c = trial.suggest_float('c', 0, 1)
    d = trial.suggest_float('d', c, 3)
    e = trial.suggest_float('e', d, 6)
    f = trial.suggest_float('f', e, 50)




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
    bins = [-np.inf, -5, 0, c, d, e, f,np.inf]
    imbalance_binned = pd.cut(imbalance_raw, bins=bins, labels=False)

    # Calculer les taux de gain par bin
    win_rates = pd.crosstab(
        imbalance_binned,
        target_y,
        normalize='index'
    )

    # Vérifier que les deux classes existent dans le crosstab
    if 1 not in win_rates.columns or win_rates.shape[0] < 2:
        return -np.inf

    # Extraire les taux de gain (proportion de classe 1)
    win_rates = win_rates[1]

    # Calculer la fréquence des bins
    bin_counts = pd.Series(imbalance_binned).value_counts(normalize=True)

    # Vérifier que chaque bin contient au moins 1% des données
    min_bin_pct = bin_counts.min() if not bin_counts.empty else 0
    if min_bin_pct < 0.01:
        return -np.inf

    # Calculer la différence entre le meilleur et le pire taux de gain
    highest_win_rate = win_rates.max()
    lowest_win_rate = win_rates.min()
    win_rate_spread = highest_win_rate - lowest_win_rate

    # Calculer l'information mutuelle pour la significativité statistique
    mi_score = mutual_info_score(imbalance_binned, target_y)

    # Test du Chi² pour la significativité statistique
    contingency = pd.crosstab(imbalance_binned, target_y)
    chi2, p_value, _, _ = chi2_contingency(contingency)
    power_stat = -np.log10(p_value + 1e-10)

    # Stocker ces métriques comme des valeurs intermédiaires pour les contraintes
    trial.set_user_attr('highest_win_rate', float(highest_win_rate))
    trial.set_user_attr('lowest_win_rate', float(lowest_win_rate))
    trial.set_user_attr('win_rate_spread', float(win_rate_spread))
    trial.set_user_attr('min_bin_pct', float(min_bin_pct))

    # Score principal: une combinaison de l'information mutuelle et du test chi²
    # On peut ajuster les poids (alpha et beta) selon l'importance relative
    alpha = 1  # Poids pour l'information mutuelle
    beta = 0  # Poids pour la puissance statistique via Chi²

    ev = evaluate_feature(imbalance_binned, target_y)

    # {
    #     'iv': iv,
    #     'mi': mi,
    #     'p_value': p_value,
    #     'combined_score': score
    # }

    return ev['combined_score']


def calculate_constraints_optuna(trial, config):
    # Récupérer les métriques stockées par l'essai
    highest_win_rate = trial.user_attrs.get('highest_win_rate', 0)
    lowest_win_rate = trial.user_attrs.get('lowest_win_rate', 0)
    min_bin_pct = trial.user_attrs.get('min_bin_pct', 0)

    constraints = []

    # Contrainte 1: Le taux de gain le plus élevé doit être supérieur à 53%
    constraints.append(highest_win_rate >= 0.53)

    # Contrainte 2: Le taux de gain le plus bas doit être inférieur à 47%
    constraints.append(lowest_win_rate <= 0.47)

    # Contrainte 3: Assurer que chaque bin contient au moins 5% des données
    constraints.append(min_bin_pct >= 0.051)

    return constraints


def create_constraints_func():
    def constraints_func(trial):
        constraints = calculate_constraints_optuna(trial=trial, config={})
        return constraints

    return constraints_func

import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard
from pynput import keyboard as pynput_keyboard  # Alternative si keyboard pose problème

# Variable globale pour contrôler l'arrêt de l'optimisation
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

# Configuration et exécution de l'optimisation avec contraintes
sampler = optuna.samplers.TPESampler(
    seed=41,
    constraints_func=create_constraints_func()
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

# Démarrer le listener de clavier avant l'optimisation
listener = pynput_keyboard.Listener(on_press=on_press)
listener.start()

# Lancer l'optimisation avec les callbacks
study.optimize(
    objective,
    n_trials=9000,
    callbacks=[callback_optuna_stop]
)

# Arrêter proprement le listener après l'optimisation
listener.stop()
listener.join()

# Meilleurs paramètres
best_params = study.best_params
print("Meilleurs paramètres trouvés:")
print(f"y = {best_params['yy']}, z = {best_params['z']}")  # Since yy is fixed at 1
print(f"Seuils de bins: c = {best_params['c']}, d = {best_params['d']}, e = {best_params['e']}"
      f", f = {best_params['f']}")

# Appliquer les meilleurs paramètres au dataset final
best_z =  best_params['z']
best_yy =  best_params['yy']
best_a, best_b = -5, 0
best_c, best_d, best_e= best_params['c'], best_params['d'], best_params['e']
best_f=best_params['f']

# Constants for special values - use the SAME values as in the objective function
Imb_Div0 = -6
Imb_zone = -2

# Create the raw feature using exactly the same logic as in the objective function
if is_bull:
    imbalance_raw = np.where(
        df_filtered[bid_vol_col] < best_params['yy'],
        Imb_Div0,
        np.where(
            (df_filtered[bid_vol_col] >= best_params['yy']) & (df_filtered[bid_vol_col] <= best_params['z']),
            Imb_zone,
            df_filtered[ask_vol_col] / df_filtered[bid_vol_col]
        )
    )
else:
    imbalance_raw = np.where(
        df_filtered[ask_vol_col] < best_params['yy'],
        Imb_Div0,
        np.where(
            (df_filtered[ask_vol_col] >= best_params['yy']) & (df_filtered[ask_vol_col] <= best_params['z']),
            Imb_zone,
            df_filtered[bid_vol_col] / df_filtered[ask_vol_col]
        )
    )

# Use the exact same binning thresholds as in the objective function
best_bins = [-np.inf, -5, 0, best_params['c'], best_params['d'], best_params['e'],best_params['f'], np.inf]
feature_name = f"{SELECTED_FEATURE}_optimized"
df_filtered.loc[:, feature_name] = pd.cut(imbalance_raw, bins=best_bins, labels=False)

# Analysis with the new feature
bin_analysis = pd.crosstab(df_filtered[feature_name], target_y, normalize='index')
bin_counts = df_filtered[feature_name].value_counts().sort_index()

analysis_df = pd.DataFrame({
    'Bin': range(len(bin_counts)),
    'Count': bin_counts.values,
    'Percentage': bin_counts.values / len(df_filtered) * 100,
    'Target_Rate': bin_analysis[1]
})

print(f"\nAnalyse détaillée des bins optimisés pour {SELECTED_FEATURE}:")
print(analysis_df)