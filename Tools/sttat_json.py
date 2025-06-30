import json
import os
import tkinter as tk
from tkinter import filedialog

def ask_file_path(initial_dir, title, filetypes):
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir, title=title, filetypes=filetypes
    )
    root.destroy()
    return file_path

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def generate_report(data, figure_path):
    report = ""

    # PHASE 1: OPTIMISATION DES PARAMÈTRES VIA VALIDATION CROISÉE TEMPORELLE
    report += "# Rapport d'Analyse XGBoost\n\n"
    report += "## PHASE 1: OPTIMISATION DES PARAMÈTRES VIA VALIDATION CROISÉE TEMPORELLE (X_train et X_val) sur l'ensemble df_train\n\n"

    ### 1. Paramètres initiaux de validations croisées
    config = data.get('config', {})
    nb_split_tscv = config.get('nb_split_tscv_', 'Non spécifié')
    cv_method = config.get('cv_method', 'Non spécifié')

    report += "### 1. Paramètres initiaux de validations croisées\n\n"
    report += f"- **Nombre de splits de la validation croisée temporelle (`nb_split_tscv_`):** {nb_split_tscv}\n"
    report += f"- **Méthode de validation croisée (`cv_method`):** {cv_method}\n"

    report += f"\n#### **Autres paramètres:**\n\n"
    early_stopping_rounds = config.get('early_stopping_rounds', 'Non spécifié')
    std_penalty_factor_ = config.get('std_penalty_factor_', 'Non spécifié')
    report += f"- **early_stopping_rounds xgb:** {early_stopping_rounds}\n"
    report += f"- **standard deviation  à soustraire au score optuna(std_penalty_factor_):** {std_penalty_factor_}\n\n"
    ### 2. Paramètres initiaux XGBoost et Optuna
    weight_param = data.get('weight_param', {})
    xgb_param_optuna_range = data.get('xgb_param_optuna_range', {})


    report += "### 2. Paramètres initiaux XGBoost avant optimisation Optuna\n\n"
    report += "#### **Paramètres de poids xgb (`weight_param`) pour fonction objective et custom_metric:**\n\n"
    for key, value in weight_param.items():
        report += f"- **{key}:**\n"
        for subkey, subvalue in value.items():
            report += f"  - {subkey}: {subvalue}\n"

    report += "\n#### **Plages des hyperparamètres XGBoost pour Optuna (`xgb_param_optuna_range`):**\n\n"
    for key, value in xgb_param_optuna_range.items():
        report += f"- **{key}:**\n"
        for subkey, subvalue in value.items():
            report += f"  - {subkey}: {subvalue}\n"

    ### 3. MEILLEURS PARAMÈTRES TROUVÉS PAR OPTUNA (sur X_TRAIN et son ensemble de validation X_VAL) PAR TYPE DE SCORE
    report += "### 3. MEILLEURS PARAMÈTRES TROUVÉS PAR OPTUNA (sur X_TRAIN et son ensemble de validation X_VAL) PAR TYPE DE SCORE\n\n"

    trials = {k: v for k, v in data.items() if k.startswith('trial_')}
    total_trials = len(trials)

    # a) Paramètres pour les trois meilleurs adjusted_score
    report += "#### a) Paramètres pour les trois meilleurs `adjusted_score`\n\n"

    # Trier les essais par adjusted_score décroissant
    sorted_trials_adjusted_score = sorted(
        trials.items(),
        key=lambda x: x[1]['best_result'].get('adjusted_score', float('-inf')),
        reverse=True
    )
    top_3_adjusted_score = sorted_trials_adjusted_score[:3]

    for idx, (trial_name, trial_data) in enumerate(top_3_adjusted_score, 1):
        best_result_trial = trial_data.get('best_result', {})
        trial_num = best_result_trial.get('current_trial_number', 'Non spécifié')
        report += f"##### **Essai numéro {idx} (Trial {trial_num})**\n\n"
        report += f"- **PnL cumulé:** {best_result_trial.get('cummulative_pnl', 'Non spécifié')}\n"
        report += f"- **Win Rate (%):** {best_result_trial.get('win_rate_percentage', 'Non spécifié')}%\n"
        report += f"- **Écart-type des scores:** {best_result_trial.get('std_dev_score', 'Non spécifié')}\n"
        report += f"- **Adjusted Score:** {best_result_trial.get('adjusted_score', 'Non spécifié')}\n"
        # Ajouter 'Scores par split' sous le format [liste]
        scores_list = best_result_trial.get('scores_ens_val_list', [])
        report += f"- **Scores par split:** {scores_list}\n\n"

        # Pour le meilleur adjusted_score (le premier), inclure les détails supplémentaires
        if idx == 1:
            # Métriques du Meilleur Essai
            report += "##### **Métriques du Meilleur Essai**\n\n"
            report += f"- **PnL cumulé:** {best_result_trial.get('cummulative_pnl', 'Non spécifié')}\n"
            report += f"- **Win Rate (%):** {best_result_trial.get('win_rate_percentage', 'Non spécifié')}%\n"
            report += f"- **Total Trades:** {best_result_trial.get('total_trades', 'Non spécifié')}\n"
            report += f"- **True Positives (TP):** {best_result_trial.get('total_tp', 'Non spécifié')}\n"
            report += f"- **False Positives (FP):** {best_result_trial.get('total_fp', 'Non spécifié')}\n"
            report += f"- **True Negatives (TN):** {best_result_trial.get('total_tn', 'Non spécifié')}\n"
            report += f"- **False Negatives (FN):** {best_result_trial.get('total_fn', 'Non spécifié')}\n"
            report += f"- **TP Percentage:** {best_result_trial.get('tp_percentage', 'Non spécifié')}%\n"
            report += f"- **Adjusted Score:** {best_result_trial.get('adjusted_score', 'Non spécifié')}\n"
            report += f"- **Écart-type des scores:** {best_result_trial.get('std_dev_score', 'Non spécifié')}\n\n"

            # Scores de Validation Croisée
            report += "##### **Scores de Validation Croisée**\n\n"
            # Afficher la liste des scores par split
            report += f"- **Scores par split:** {scores_list}\n"
            mean_score = sum(scores_list) / len(scores_list) if scores_list else 'Non spécifié'
            std_dev_score = best_result_trial.get('std_dev_score', 'Non spécifié')
            report += f"- **Score moyen:** {mean_score}\n"
            report += f"- **Écart-type des scores:** {std_dev_score}\n\n"

            # Paramètres de Trading Optimisés
            report += "##### **Paramètres de Trading Optimisés**\n\n"
            best_trial_params = trial_data.get('params', {})
            for key, value in best_trial_params.items():
                report += f"- **{key}:** {value}\n"

            report += "\n"

    # b) Résultats des 3 meilleurs PnL cumulé
    report += "#### b) Résultats des 3 meilleurs `PnL cumulé`\n\n"
    # Tri des essais par ordre décroissant de PnL cumulé
    sorted_trials_pnl = sorted(
        trials.items(),
        key=lambda x: x[1]['best_result'].get('cummulative_pnl', float('-inf')),
        reverse=True
    )
    top_3_pnl = sorted_trials_pnl[:3]
    for idx, (trial_name, trial_data) in enumerate(top_3_pnl, 1):
        best_result_trial = trial_data.get('best_result', {})
        trial_num = best_result_trial.get('current_trial_number', 'Non spécifié')
        report += f"**Essai numéro {idx} (Trial {trial_num})**\n"
        report += f"- **PnL cumulé:** {best_result_trial.get('cummulative_pnl', 'Non spécifié')}\n"
        report += f"- **Win Rate (%):** {best_result_trial.get('win_rate_percentage', 'Non spécifié')}%\n"
        report += f"- **Écart-type des scores:** {best_result_trial.get('std_dev_score', 'Non spécifié')}\n"
        report += f"- **Adjusted Score:** {best_result_trial.get('adjusted_score', 'Non spécifié')}\n"
        # Ajouter 'Scores par split' sous le format [liste]
        scores_list = best_result_trial.get('scores_ens_val_list', [])
        report += f"- **Scores par split:** {scores_list}\n\n"

    # c) Résultats des 5 meilleurs PnL cumulé triés par Écart-type des scores croissant
    report += "#### c) Résultats des 5 meilleurs `PnL cumulé` triés par `Écart-type des scores` croissant\n\n"

    # Prendre les 5 meilleurs PnL cumulé
    top_5_pnl = sorted_trials_pnl[:5]
    # Trier ces 5 essais par Écart-type des scores croissant
    top_5_pnl_sorted_by_std = sorted(
        top_5_pnl,
        key=lambda x: x[1]['best_result'].get('std_dev_score', float('inf'))
    )

    for idx, (trial_name, trial_data) in enumerate(top_5_pnl_sorted_by_std, 1):
        best_result_trial = trial_data.get('best_result', {})
        trial_num = best_result_trial.get('current_trial_number', 'Non spécifié')
        report += f"**Essai numéro {idx} (Trial {trial_num})**\n"
        report += f"- **PnL cumulé:** {best_result_trial.get('cummulative_pnl', 'Non spécifié')}\n"
        report += f"- **Win Rate (%):** {best_result_trial.get('win_rate_percentage', 'Non spécifié')}%\n"
        report += f"- **Écart-type des scores:** {best_result_trial.get('std_dev_score', 'Non spécifié')}\n"
        report += f"- **Adjusted Score:** {best_result_trial.get('adjusted_score', 'Non spécifié')}\n"
        # Ajouter 'Scores par split' sous le format [liste]
        scores_list = best_result_trial.get('scores_ens_val_list', [])
        report += f"- **Scores par split:** {scores_list}\n\n"

    # d) Résultats des 3 meilleurs Win Rate (%)
    report += "#### d) Résultats des 10 meilleurs `Win Rate (%)`\n\n"
    # Tri des essais par ordre décroissant de Win Rate (%)
    sorted_trials_winrate = sorted(
        trials.items(),
        key=lambda x: x[1]['best_result'].get('win_rate_percentage', 0),
        reverse=True
    )
    top_3_winrate = sorted_trials_winrate[:10]
    for idx, (trial_name, trial_data) in enumerate(top_3_winrate, 1):
        best_result_trial = trial_data.get('best_result', {})
        trial_num = best_result_trial.get('current_trial_number', 'Non spécifié')
        report += f"**Essai numéro {idx} (Trial {trial_num})**\n"
        report += f"- **PnL cumulé:** {best_result_trial.get('cummulative_pnl', 'Non spécifié')}\n"
        report += f"- **Win Rate (%):** {best_result_trial.get('win_rate_percentage', 'Non spécifié')}%\n"
        report += f"- **Écart-type des scores:** {best_result_trial.get('std_dev_score', 'Non spécifié')}\n"
        report += f"- **Adjusted Score:** {best_result_trial.get('adjusted_score', 'Non spécifié')}\n"
        # Ajouter 'Scores par split' sous le format [liste]
        scores_list = best_result_trial.get('scores_ens_val_list', [])
        report += f"- **Scores par split:** {scores_list}\n\n"

    # PHASE 2: ENTRAÎNEMENT FINAL
    report += "## PHASE 2: ENTRAÎNEMENT FINAL (X_train complet et ensemble de validation nouveau X_test)\n\n"
    report += "### 5. MÉTRIQUES D'ENTRAÎNEMENT FINAL\n\n"
    report += "Les métriques d'entraînement final ne sont pas disponibles dans les données fournies.\n"
    report += "Veuillez les fournir dans le prompt à ChatGPT pour une analyse complète.\n\n"

    report += "### 6. COURBES D'APPRENTISSAGE\n\n"
    report += f"La figure sélectionnée représente les courbes d'apprentissage XGBoost sur le modèle final avec les paramètres optimaux trouvés par Optuna.\n\n"
    report += "Cette figure inclut:\n"
    report += "1. Résultats sur X_train et X_train normalisé\n"
    report += "2. Résultats sur la métrique d'évaluation basée sur X_test avec normalisation de la figure\n"
    report += "3. Un zoom sur la partie avant l'early stopping\n\n"
    report += "Veuillez fournir des détails supplémentaires sur ces résultats dans le prompt à ChatGPT pour une analyse approfondie.\n\n"

    # CRITÈRES DE RÉFÉRENCE POUR L'ANALYSE DES COURBES
    report += "## CRITÈRES DE RÉFÉRENCE POUR L'ANALYSE DES COURBES\n\n"
    report += "### Courbe X_train (bleue) - Caractéristiques Optimales\n"
    report += "1. Progression régulière mais pas trop rapide\n"
    report += "2. Pente modérée (pas verticale)\n"
    report += "3. Légère courbure (forme concave) indiquant un apprentissage progressif\n"
    report += "4. Potentiellement un léger plateau vers la fin, mais pas plat trop tôt\n\n"
    report += "### Courbe X_test (rouge) - Caractéristiques Optimales\n"
    report += "1. Progression similaire à X_train avec :\n"
    report += "   - Score plus bas que X_train (écart raisonnable)\n"
    report += "   - Oscillations minimales\n"
    report += "   - Tendance haussière stable jusqu'à l'early stopping\n"
    report += "2. Absence de chute brutale après le pic\n"
    report += "3. Convergence visible vers un niveau stable\n\n"
    report += "Veuillez fournir les courbes ou les données correspondantes pour une analyse détaillée.\n\n"

    # STRUCTURE DU RAPPORT D'ANALYSE
    report += "## STRUCTURE DU RAPPORT D'ANALYSE\n\n"
    report += "1. **Analyse des Paramètres Optimisés**\n"
    report += "   - Évaluation des paramètres par rapport aux plages définies\n"
    report += "   - Identification des paramètres proches des bornes\n"
    report += "   - Analyse de la robustesse des paramètres (via écart-type des scores)\n\n"
    report += "2. **Analyse des Performances de Trading**\n"
    report += "   - Évaluation des métriques de trading (validation)\n"
    report += "   - Comparaison des performances train/test\n"
    report += "   - Analyse du compromis précision/rappel\n\n"
    report += "3. **Analyse des Courbes d'Entraînement Final**\n"
    report += "   - Évaluation de la progression sur X_train\n"
    report += "   - Analyse de la stabilité sur X_test\n"
    report += "   - Évaluation de l'early stopping\n"
    report += "   - Diagnostic du surapprentissage\n\n"
    report += "4. **Diagnostic Global**\n"
    report += "   - Points forts du modèle\n"
    report += "   - Points d'attention\n"
    report += "   - Risques identifiés\n\n"
    report += "5. **Recommandations**\n"
    report += "   - Ajustements proposés pour les paramètres\n"
    report += "   - Modifications suggérées pour les plages Optuna\n"
    report += "   - Suggestions pour la stratégie d'entraînement\n\n"
    report += "6. **Plan d'Action**\n"
    report += "   - Prochaines étapes prioritaires\n"
    report += "   - Tests suggérés\n"
    report += "   - Modifications à valider\n\n"

    return report

def main():
    # Demander l'emplacement du fichier JSON
    initial_dir_json = r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\results_optim"
    json_path = ask_file_path(
        initial_dir=initial_dir_json,
        title="Sélectionnez le fichier JSON",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    """"
    # Demander l'emplacement de la figure représentant les tests
    initial_dir_fig = initial_dir_json  # Même répertoire que le JSON
    figure_path = ask_file_path(
        initial_dir=initial_dir_fig,
        title="Sélectionnez la figure représentant les courbes d'apprentissage",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]
    )

    # Demander l'emplacement des 2 fichiers Excel pour l'analyse SHAP
    initial_dir_excel = initial_dir_json  # Vous pouvez changer si nécessaire
    shap_train_path = ask_file_path(
        initial_dir=initial_dir_excel,
        title="Sélectionnez le fichier Excel SHAP pour l'entraînement",
        filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    shap_test_path = ask_file_path(
        initial_dir=initial_dir_excel,
        title="Sélectionnez le fichier Excel SHAP pour le test",
        filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    """
    # Charger le fichier JSON
    if json_path and os.path.exists(json_path):
        data = load_json(json_path)
        # Générer le rapport
        report = generate_report(data, json_path)
        # Afficher ou sauvegarder le rapport
        print(report)
        # Vous pouvez également enregistrer le rapport dans un fichier
        with open('rapport_analyse_xgboost.md', 'w', encoding='utf-8') as f:
            f.write(report)
    else:
        print("Le fichier JSON n'a pas été sélectionné ou n'existe pas.")

if __name__ == "__main__":
    main()
