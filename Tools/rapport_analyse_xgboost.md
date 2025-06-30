# Rapport d'Analyse XGBoost

## PHASE 1: OPTIMISATION DES PARAMÈTRES VIA VALIDATION CROISÉE TEMPORELLE (X_train et X_val) sur l'ensemble df_train

### 1. Paramètres initiaux de validations croisées

- **Nombre de splits de la validation croisée temporelle (`nb_split_tscv_`):** 5
- **Méthode de validation croisée (`cv_method`):** cv_config.K_FOLD

#### **Autres paramètres:**

- **early_stopping_rounds xgb:** 50
- **standard deviation  à soustraire au score optuna(std_penalty_factor_):** Non spécifié

### 2. Paramètres initiaux XGBoost avant optimisation Optuna

#### **Paramètres de poids xgb (`weight_param`) pour fonction objective et custom_metric:**

- **threshold:**
  - min: 0.6
  - max: 0.77
- **w_p:**
  - min: 1.2
  - max: 2.5
- **w_n:**
  - min: 1
  - max: 1
- **profit_per_tp:**
  - min: 1.5
  - max: 1.5
- **loss_per_fp:**
  - min: -1.1
  - max: -1.1
- **penalty_per_fn:**
  - min: -0.0
  - max: -0.0
- **weight_split:**
  - min: 0.65
  - max: 0.65
- **nb_split_weight:**
  - min: 0
  - max: 0
- **std_penalty_factor:**
  - min: 0.9
  - max: 1.1

#### **Plages des hyperparamètres XGBoost pour Optuna (`xgb_param_optuna_range`):**

- **num_boost_round:**
  - min: 200
  - max: 900
- **max_depth:**
  - min: 3
  - max: 8
- **learning_rate:**
  - min: 0.005
  - max: 0.2
  - log: True
- **min_child_weight:**
  - min: 1
  - max: 5
- **subsample:**
  - min: 0.6
  - max: 0.8
- **colsample_bytree:**
  - min: 0.55
  - max: 0.99
- **colsample_bylevel:**
  - min: 0.6
  - max: 0.9
- **colsample_bynode:**
  - min: 0.5
  - max: 0.85
- **gamma:**
  - min: 0
  - max: 10
- **reg_alpha:**
  - min: 0.1
  - max: 9
  - log: True
- **reg_lambda:**
  - min: 0.1
  - max: 9
  - log: True
### 3. MEILLEURS PARAMÈTRES TROUVÉS PAR OPTUNA (sur X_TRAIN et son ensemble de validation X_VAL) PAR TYPE DE SCORE

#### a) Paramètres pour les trois meilleurs `adjusted_score`

##### **Essai numéro 1 (Trial 1)**

- **PnL cumulé:** -2924.5
- **Win Rate (%):** 41.26%
- **Écart-type des scores:** 781.2159368574096
- **Adjusted Score:** Non spécifié
- **Scores par split:** [-995.4, -1100.7, -221.0, 622.6, -1230.0]

##### **Métriques du Meilleur Essai**

- **PnL cumulé:** -2924.5
- **Win Rate (%):** 41.26%
- **Total Trades:** Non spécifié
- **True Positives (TP):** Non spécifié
- **False Positives (FP):** Non spécifié
- **True Negatives (TN):** Non spécifié
- **False Negatives (FN):** Non spécifié
- **TP Percentage:** 19.789%
- **Adjusted Score:** Non spécifié
- **Écart-type des scores:** 781.2159368574096

##### **Scores de Validation Croisée**

- **Scores par split:** [-995.4, -1100.7, -221.0, 622.6, -1230.0]
- **Score moyen:** -584.9
- **Écart-type des scores:** 781.2159368574096

##### **Paramètres de Trading Optimisés**

- **max_depth:** 5
- **learning_rate:** 0.1667521176194013
- **min_child_weight:** 4
- **subsample:** 0.7197316968394073
- **colsample_bytree:** 0.6186482017946721
- **colsample_bylevel:** 0.6467983561008608
- **colsample_bynode:** 0.5203292642588698
- **gamma:** 8.661761457749352
- **reg_alpha:** 1.4952868327767481
- **reg_lambda:** 2.4196108945582093
- **threshold:** 0.6034993640302864
- **profit_per_tp:** 1.5
- **loss_per_fp:** -1.1
- **penalty_per_fn:** -0.0
- **num_boost_round:** 879
- **w_p:** 2.282175433040548
- **w_n:** 1.0
- **weight_split:** 0.65
- **nb_split_weight:** 0
- **std_penalty_factor:** 0.9424678221356553

##### **Essai numéro 2 (Trial 2)**

- **PnL cumulé:** 0.0
- **Win Rate (%):** 0%
- **Écart-type des scores:** 0.0
- **Adjusted Score:** Non spécifié
- **Scores par split:** [0.0, 0.0, 0.0, 0.0, 0.0]

##### **Essai numéro 3 (Trial 3)**

- **PnL cumulé:** 0.0
- **Win Rate (%):** 0%
- **Écart-type des scores:** 0.0
- **Adjusted Score:** Non spécifié
- **Scores par split:** [0.0, 0.0, 0.0, 0.0, 0.0]

#### b) Résultats des 3 meilleurs `PnL cumulé`

**Essai numéro 1 (Trial 131)**
- **PnL cumulé:** 1219.4999999999964
- **Win Rate (%):** 43.49%
- **Écart-type des scores:** 285.54155564470824
- **Adjusted Score:** Non spécifié
- **Scores par split:** [3.7, 98.3, 378.7, 683.7, 55.1]

**Essai numéro 2 (Trial 52)**
- **PnL cumulé:** 1164.5999999999985
- **Win Rate (%):** 43.52%
- **Écart-type des scores:** 382.89511226966584
- **Adjusted Score:** Non spécifié
- **Scores par split:** [-9.4, 110.4, 202.9, 896.1, -35.4]

**Essai numéro 3 (Trial 105)**
- **PnL cumulé:** 1053.8999999999996
- **Win Rate (%):** 45.63%
- **Écart-type des scores:** 208.77845195326074
- **Adjusted Score:** Non spécifié
- **Scores par split:** [45.2, 111.8, 293.5, 540.2, 63.2]

#### c) Résultats des 5 meilleurs `PnL cumulé` triés par `Écart-type des scores` croissant

**Essai numéro 1 (Trial 105)**
- **PnL cumulé:** 1053.8999999999996
- **Win Rate (%):** 45.63%
- **Écart-type des scores:** 208.77845195326074
- **Adjusted Score:** Non spécifié
- **Scores par split:** [45.2, 111.8, 293.5, 540.2, 63.2]

**Essai numéro 2 (Trial 121)**
- **PnL cumulé:** 1020.6999999999998
- **Win Rate (%):** 45.58%
- **Écart-type des scores:** 221.23896130654748
- **Adjusted Score:** Non spécifié
- **Scores par split:** [46.0, 102.4, 285.0, 556.2, 31.1]

**Essai numéro 3 (Trial 96)**
- **PnL cumulé:** 1052.199999999999
- **Win Rate (%):** 45.14%
- **Écart-type des scores:** 230.5334531038825
- **Adjusted Score:** Non spécifié
- **Scores par split:** [66.8, 44.7, 342.4, 555.4, 42.9]

**Essai numéro 4 (Trial 131)**
- **PnL cumulé:** 1219.4999999999964
- **Win Rate (%):** 43.49%
- **Écart-type des scores:** 285.54155564470824
- **Adjusted Score:** Non spécifié
- **Scores par split:** [3.7, 98.3, 378.7, 683.7, 55.1]

**Essai numéro 5 (Trial 52)**
- **PnL cumulé:** 1164.5999999999985
- **Win Rate (%):** 43.52%
- **Écart-type des scores:** 382.89511226966584
- **Adjusted Score:** Non spécifié
- **Scores par split:** [-9.4, 110.4, 202.9, 896.1, -35.4]

#### d) Résultats des 10 meilleurs `Win Rate (%)`

**Essai numéro 1 (Trial 14)**
- **PnL cumulé:** 1.5
- **Win Rate (%):** 100.0%
- **Écart-type des scores:** 0.6708203932499369
- **Adjusted Score:** Non spécifié
- **Scores par split:** [1.5, 0.0, 0.0, 0.0, 0.0]

**Essai numéro 2 (Trial 19)**
- **PnL cumulé:** 3.0
- **Win Rate (%):** 100.0%
- **Écart-type des scores:** 1.3416407864998738
- **Adjusted Score:** Non spécifié
- **Scores par split:** [0.0, 0.0, 3.0, 0.0, 0.0]

**Essai numéro 3 (Trial 12)**
- **PnL cumulé:** 20.799999999999997
- **Win Rate (%):** 73.08%
- **Écart-type des scores:** 4.0041228752374725
- **Adjusted Score:** Non spécifié
- **Scores par split:** [8.3, 7.6, 0.0, 0.0, 4.9]

**Essai numéro 4 (Trial 189)**
- **PnL cumulé:** 9.1
- **Win Rate (%):** 69.23%
- **Écart-type des scores:** 2.1147103820618085
- **Adjusted Score:** Non spécifié
- **Scores par split:** [3.0, 0.0, 1.2, 0.0, 4.9]

**Essai numéro 5 (Trial 171)**
- **PnL cumulé:** 133.1
- **Win Rate (%):** 59.66%
- **Écart-type des scores:** 11.285255867724045
- **Adjusted Score:** Non spécifié
- **Scores par split:** [25.2, 27.5, 45.1, 16.2, 19.1]

**Essai numéro 6 (Trial 68)**
- **PnL cumulé:** 137.6
- **Win Rate (%):** 58.64%
- **Écart-type des scores:** 10.731588885155823
- **Adjusted Score:** Non spécifié
- **Scores par split:** [43.0, 20.6, 34.6, 20.3, 19.1]

**Essai numéro 7 (Trial 25)**
- **PnL cumulé:** 12.299999999999999
- **Win Rate (%):** 58.62%
- **Écart-type des scores:** 4.038316480911322
- **Adjusted Score:** Non spécifié
- **Scores par split:** [9.3, 3.0, 0.0, 0.0, 0.0]

**Essai numéro 8 (Trial 60)**
- **PnL cumulé:** 108.39999999999999
- **Win Rate (%):** 58.59%
- **Écart-type des scores:** 13.503592114693038
- **Adjusted Score:** Non spécifié
- **Scores par split:** [28.2, 22.1, 36.4, 0.0, 21.7]

**Essai numéro 9 (Trial 169)**
- **PnL cumulé:** 156.1
- **Win Rate (%):** 58.49%
- **Écart-type des scores:** 12.998153715047382
- **Adjusted Score:** Non spécifié
- **Scores par split:** [20.7, 30.7, 53.1, 29.5, 22.1]

**Essai numéro 10 (Trial 44)**
- **PnL cumulé:** 77.1
- **Win Rate (%):** 57.36%
- **Écart-type des scores:** 12.355848817462926
- **Adjusted Score:** Non spécifié
- **Scores par split:** [28.4, 8.0, 28.5, 1.5, 10.7]

## PHASE 2: ENTRAÎNEMENT FINAL (X_train complet et ensemble de validation nouveau X_test)

### 5. MÉTRIQUES D'ENTRAÎNEMENT FINAL

Les métriques d'entraînement final ne sont pas disponibles dans les données fournies.
Veuillez les fournir dans le prompt à ChatGPT pour une analyse complète.

### 6. COURBES D'APPRENTISSAGE

La figure sélectionnée représente les courbes d'apprentissage XGBoost sur le modèle final avec les paramètres optimaux trouvés par Optuna.

Cette figure inclut:
1. Résultats sur X_train et X_train normalisé
2. Résultats sur la métrique d'évaluation basée sur X_test avec normalisation de la figure
3. Un zoom sur la partie avant l'early stopping

Veuillez fournir des détails supplémentaires sur ces résultats dans le prompt à ChatGPT pour une analyse approfondie.

## CRITÈRES DE RÉFÉRENCE POUR L'ANALYSE DES COURBES

### Courbe X_train (bleue) - Caractéristiques Optimales
1. Progression régulière mais pas trop rapide
2. Pente modérée (pas verticale)
3. Légère courbure (forme concave) indiquant un apprentissage progressif
4. Potentiellement un léger plateau vers la fin, mais pas plat trop tôt

### Courbe X_test (rouge) - Caractéristiques Optimales
1. Progression similaire à X_train avec :
   - Score plus bas que X_train (écart raisonnable)
   - Oscillations minimales
   - Tendance haussière stable jusqu'à l'early stopping
2. Absence de chute brutale après le pic
3. Convergence visible vers un niveau stable

Veuillez fournir les courbes ou les données correspondantes pour une analyse détaillée.

## STRUCTURE DU RAPPORT D'ANALYSE

1. **Analyse des Paramètres Optimisés**
   - Évaluation des paramètres par rapport aux plages définies
   - Identification des paramètres proches des bornes
   - Analyse de la robustesse des paramètres (via écart-type des scores)

2. **Analyse des Performances de Trading**
   - Évaluation des métriques de trading (validation)
   - Comparaison des performances train/test
   - Analyse du compromis précision/rappel

3. **Analyse des Courbes d'Entraînement Final**
   - Évaluation de la progression sur X_train
   - Analyse de la stabilité sur X_test
   - Évaluation de l'early stopping
   - Diagnostic du surapprentissage

4. **Diagnostic Global**
   - Points forts du modèle
   - Points d'attention
   - Risques identifiés

5. **Recommandations**
   - Ajustements proposés pour les paramètres
   - Modifications suggérées pour les plages Optuna
   - Suggestions pour la stratégie d'entraînement

6. **Plan d'Action**
   - Prochaines étapes prioritaires
   - Tests suggérés
   - Modifications à valider

