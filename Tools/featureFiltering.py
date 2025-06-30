from standardFunc_sauv import (load_data, split_sessions, print_notification,
                               plot_calibrationCurve_distrib, plot_fp_tp_rates, check_gpu_availability,
                               timestamp_to_date_utc, calculate_and_display_sessions,
                               timestamp_to_date_utc, calculate_and_display_sessions,
                               calculate_weighted_adjusted_score_custom, sigmoidCustom,
                               custom_metric_ProfitBased_cpu, create_weighted_logistic_obj_cpu,
                               train_finalModel_analyse, init_dataSet, sigmoidCustom_cpu)
import pandas as pd
import os

# Configuration pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
def select_features(correlation_df, shap_importance_df_train, shap_importance_df_test, interaction_df,
                    top_n=30, correlation_threshold=0.9, interaction_threshold=10.0,
                    shap_percentage=0.80):
    """
    Sélection de features avec débogage détaillé
    """
    # 1. Debug du chargement initial
    print("\n=== DEBUG: Données SHAP initiales ===")
    print("Train SHAP head:")
    print(shap_importance_df_train.head())
    print("\nTrain SHAP shape:", shap_importance_df_train.shape)

    def prepare_dataframes():
        print("\n=== DEBUG: Préparation des DataFrames ===")

        # Conversion et vérification des valeurs SHAP
        initial_shap_values = shap_importance_df_train['importance'].head()
        print("Valeurs SHAP avant conversion:", initial_shap_values)

        # Conversion des colonnes en numérique avec débogage
        shap_importance_df_train['importance'] = pd.to_numeric(
            shap_importance_df_train['importance'], errors='coerce')
        shap_importance_df_test['importance'] = pd.to_numeric(
            shap_importance_df_test['importance'], errors='coerce')

        print("Valeurs SHAP après conversion:", shap_importance_df_train['importance'].head())

        correlation_df[['feature1', 'feature2']] = correlation_df['Correlation_Pair'].str.extract(
            r'(.+) <-> (.+)')
        correlation_df['Correlation_Value'] = pd.to_numeric(
            correlation_df['Correlation_Value'], errors='coerce')

        interaction_df[['feature1', 'feature2']] = interaction_df['Interaction'].str.extract(
            r'(.+) <-> (.+)')
        interaction_df['Value'] = pd.to_numeric(
            interaction_df['Value'], errors='coerce')

        # Vérifier les valeurs NaN
        print("\nNombre de valeurs NaN:")
        print("SHAP train:", shap_importance_df_train['importance'].isna().sum())
        print("SHAP test:", shap_importance_df_test['importance'].isna().sum())
        print("Corrélations:", correlation_df['Correlation_Value'].isna().sum())
        print("Interactions:", interaction_df['Value'].isna().sum())

        return correlation_df, shap_importance_df_train, shap_importance_df_test, interaction_df

    def get_top_features(shap_df, n):
        """
        Sélection des features basée sur l'importance SHAP absolue
        """
        print("\n=== DEBUG: Sélection initiale des features ===")

        # Conversion en valeurs absolues
        shap_df = shap_df.copy()
        shap_df['abs_importance'] = shap_df['importance'].abs()

        # Afficher les valeurs avant tri
        print("Top 5 avant tri:")
        print(shap_df[['feature', 'importance', 'abs_importance']].head())

        # Trier par importance absolue décroissante
        sorted_df = shap_df.sort_values('abs_importance', ascending=False)

        print("\nTop 5 après tri (vérification des valeurs absolues):")
        top_5 = sorted_df.head()
        for _, row in top_5.iterrows():
            print(f"{row['feature']}: SHAP={row['importance']}, |SHAP|={row['abs_importance']}")

        # Vérification supplémentaire
        print("\nTop 10 features par ordre d'importance absolue:")
        for idx, (feature, abs_imp, imp) in enumerate(
                sorted_df[['feature', 'abs_importance', 'importance']].values[:10], 1):
            print(f"{idx}. {feature}: |SHAP|={abs_imp:.6f}, SHAP={imp:.6f}")

        return sorted_df['feature'].head(n).tolist()

    def test_correlation_logic():
        """Test de la logique de sélection des features corrélées"""
        print("\n=== TEST LOGIQUE CORRELATION ===")

        # Cas de test
        test_cases = [
            # (feature1, shap1, feature2, shap2, expected_kept)
            ("A", -99, "B", -10, "A"),  # Test 1: négatif plus grand vs négatif plus petit
            ("A", 10, "B", -99, "B"),  # Test 2: positif vs négatif plus grand
            ("A", -50, "B", 60, "B"),  # Test 3: négatif vs positif plus grand
            ("A", -0.5, "B", 0.1, "A"),  # Test 4: petites valeurs
        ]

        for f1, s1, f2, s2, expected in test_cases:
            imp1 = abs(s1)
            imp2 = abs(s2)

            kept = f1 if imp1 >= imp2 else f2
            removed = f2 if imp1 >= imp2 else f1

            print(f"\nTest avec {f1}({s1}) et {f2}({s2}):")
            print(f"Valeurs absolues: |{s1}| = {imp1}, |{s2}| = {imp2}")
            print(f"Feature conservée: {kept} (attendu: {expected})")
            print(f"Feature supprimée: {removed}")
            print(f"Résultat correct: {kept == expected}")

    test_correlation_logic()

    # Fonction corrigée pour handle_correlations
    def handle_correlations(corr_df, top_feats, shap_dict):
        """
        Version corrigée du traitement des corrélations basée sur les valeurs absolues SHAP
        """
        print("\n=== DEBUG: Traitement des corrélations ===")
        features_to_remove = set()
        correlation_info = {}

        correlated_pairs = corr_df[
            (corr_df['feature1'].isin(top_feats)) &
            (corr_df['feature2'].isin(top_feats)) &
            (corr_df['Correlation_Value'].abs() >= correlation_threshold)
            ]

        print(f"Nombre de paires corrélées trouvées: {len(correlated_pairs)}")

        for _, row in correlated_pairs.iterrows():
            f1, f2 = row['feature1'], row['feature2']
            s1 = shap_dict.get(f1, 0)
            s2 = shap_dict.get(f2, 0)
            imp1 = abs(s1)
            imp2 = abs(s2)

            # Debug détaillé
            print(f"\nAnalyse de la paire corrélée: {f1} <-> {f2}")
            print(f"{f1}: SHAP = {s1} (|SHAP| = {imp1})")
            print(f"{f2}: SHAP = {s2} (|SHAP| = {imp2})")
            print(f"Corrélation: {row['Correlation_Value']}")

            # Sélection basée sur la valeur absolue la plus élevée
            if imp1 >= imp2:
                kept_feature = f1
                removed_feature = f2
            else:
                kept_feature = f2
                removed_feature = f1

            print(f"Décision: conservation de {kept_feature} (|SHAP| = {max(imp1, imp2)})")
            print(f"Suppression de {removed_feature} (|SHAP| = {min(imp1, imp2)})")

            features_to_remove.add(removed_feature)
            correlation_info[removed_feature] = {
                'correlated_with': kept_feature,
                'correlation_value': row['Correlation_Value'],
                'removed_shap': shap_dict.get(removed_feature, 0),
                'kept_shap': shap_dict.get(kept_feature, 0)
            }

        print(f"\nRésumé des suppressions par corrélation:")
        for removed in features_to_remove:
            info = correlation_info[removed]
            print(f"{removed} (SHAP={info['removed_shap']}) supprimée, "
                  f"corrélée avec {info['correlated_with']} (SHAP={info['kept_shap']})")

        return features_to_remove, correlation_info

    def handle_interactions(inter_df, selected_feats, removed_feats):
        print("\n=== DEBUG: Traitement des interactions ===")
        features_to_add = set()
        interaction_info = {}

        important_interactions = inter_df[
            inter_df['Value'] >= interaction_threshold
            ]

        print(f"Nombre d'interactions importantes trouvées: {len(important_interactions)}")

        for _, row in important_interactions.iterrows():
            f1, f2 = row['feature1'], row['feature2']
            if (f1 in removed_feats and f2 in selected_feats):
                features_to_add.add(f1)
                interaction_info[f1] = {
                    'interacts_with': f2,
                    'interaction_value': row['Value']
                }
            elif (f2 in removed_feats and f1 in selected_feats):
                features_to_add.add(f2)
                interaction_info[f2] = {
                    'interacts_with': f1,
                    'interaction_value': row['Value']
                }

        print(f"Features à réintégrer pour interactions: {features_to_add}")
        return features_to_add, interaction_info

    def handle_stability(shap_df_train, shap_df_test, features, user_choice, shap_dict):
        print("\n=== DEBUG: Traitement de la stabilité ===")
        stability_info = {}
        features_to_remove_stability = set()

        shap_train = shap_df_train.set_index('feature')['importance'].to_dict()
        shap_test = shap_df_test.set_index('feature')['importance'].to_dict()

        print("Analyse de stabilité pour les premières features:")
        for feature in list(features)[:5]:
            impact_train = shap_train.get(feature, 0)
            impact_test = shap_test.get(feature, 0)
            print(f"\nFeature: {feature}")
            print(f"Impact train: {impact_train}")
            print(f"Impact test: {impact_test}")

            sign_train = 'positive' if impact_train >= 0 else 'negative'
            sign_test = 'positive' if impact_test >= 0 else 'negative'
            is_stable = sign_train == sign_test

            stability_info[feature] = {
                'train_impact': impact_train,
                'test_impact': impact_test,
                'is_stable': is_stable,
                'impact_sign': sign_train if is_stable else 'unstable'
            }

            if not is_stable:
                features_to_remove_stability.add(feature)

        print(f"\nFeatures instables à supprimer: {features_to_remove_stability}")
        return features_to_remove_stability, stability_info

    try:
        correlation_df, shap_importance_df_train, shap_importance_df_test, interaction_df = prepare_dataframes()

        print("\nOptions pour la prise en compte de la stabilité des features :")
        print("Appuyez sur 'Entrée' pour réintégrer toutes les features stables (impact positif ou négatif).")
        print("Tapez 'p' pour réintégrer uniquement les features stables avec impact positif.")
        print("Tapez 'n' pour réintégrer uniquement les features stables avec impact négatif.")
        print("Appuyez sur 'x' pour ignorer la stabilité des features.")
        user_input = input("Votre choix : ").strip().lower()

        if user_input == ('x'):
            print("\n=== DEBUG: Stabilité ignorée ===")
            consider_stability = False
            user_choice = None
            print("La stabilité des features ne sera pas prise en compte.")
        else:
            consider_stability = True
            user_choice = user_input if user_input in ['p', 'n'] else 'all'
            print("La stabilité des features sera prise en compte selon votre choix.")

        top_features = get_top_features(shap_importance_df_train, top_n)
        shap_dict = shap_importance_df_train.set_index('feature')['importance'].to_dict()

        print("\n=== DEBUG: SHAP Dictionary ===")
        print("Nombre de features dans shap_dict:", len(shap_dict))
        print("Top 5 valeurs SHAP:")
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, value in sorted_shap:
            print(f"{feature}: {value}")

        features_to_remove_corr, correlation_info = handle_correlations(
            correlation_df, top_features, shap_dict)
        selected_features = [f for f in top_features if f not in features_to_remove_corr]

        features_to_add, interaction_info = handle_interactions(
            interaction_df, selected_features, features_to_remove_corr)

        selected_features.extend(features_to_add)
        features_to_remove_corr -= features_to_add

        if consider_stability:
            cumulative_importance = 0
            abs_shap_values = {f: abs(shap_dict.get(f, 0)) for f in selected_features}
            total_importance = sum(abs_shap_values.values())
            threshold_importance = shap_percentage * total_importance

            selected_features_above_threshold = []
            selected_features_below_threshold = []

            for feature in selected_features:
                cumulative_importance += abs_shap_values.get(feature, 0)
                if cumulative_importance <= threshold_importance:
                    selected_features_above_threshold.append(feature)
                else:
                    selected_features_below_threshold.append(feature)

            features_to_remove_stability_above, stability_info_above = handle_stability(
                shap_importance_df_train, shap_importance_df_test,
                selected_features_above_threshold, user_choice, shap_dict)

            features_to_remove_stability_below, stability_info_below = handle_stability(
                shap_importance_df_train, shap_importance_df_test,
                selected_features_below_threshold, user_choice, shap_dict)

            selected_features = [
                f for f in selected_features_above_threshold
                if f not in features_to_remove_stability_above
            ]

            if user_choice in ['p', 'n']:
                for feature in selected_features_below_threshold:
                    if feature not in features_to_remove_stability_below:
                        shap_value = shap_dict.get(feature, 0)
                        if (user_choice == 'p' and shap_value > 0) or \
                                (user_choice == 'n' and shap_value < 0):
                            selected_features.append(feature)
            else:
                selected_features.extend([
                    f for f in selected_features_below_threshold
                    if f not in features_to_remove_stability_below
                ])

            stability_info = {**stability_info_above, **stability_info_below}

        else:
            stability_info = {}

        sorted_selected = sorted(
            [(f, abs(shap_dict.get(f, 0)))
             for f in selected_features],
            key=lambda x: x[1],
            reverse=True
        )

        selected_details = {}
        for feature, abs_shap_value in sorted_selected:
            shap_value = shap_dict.get(feature, 0)
            details = {
                'shap_value': shap_value,
                'abs_shap_value': abs_shap_value,
                'readded_via_interactions': feature in features_to_add
            }
            if feature in interaction_info:
                details.update(interaction_info[feature])
            if feature in stability_info:
                details.update(stability_info[feature])
            selected_details[feature] = details

        features_removed = set(features_to_remove_corr)
        if consider_stability:
            features_removed.update(features_to_remove_stability_above)
            features_removed.update(features_to_remove_stability_below)

        sorted_removed = sorted(
            [(f, abs(shap_dict.get(f, 0))) for f in features_removed],
            key=lambda x: x[1],
            reverse=True
        )

        removed_details = {}
        for feature, abs_shap_value in sorted_removed:
            shap_value = shap_dict.get(feature, 0)
            details = {
                'shap_value': shap_value,
                'abs_shap_value': abs_shap_value,
                **correlation_info.get(feature, {})
            }
            if feature in stability_info:
                details.update(stability_info[feature])
            removed_details[feature] = details

        return {
            'selected_features': [f[0] for f in sorted_selected],
            'selected_features_with_details': selected_details,
            'removed_features': [f[0] for f in sorted_removed],
            'removed_features_with_details': removed_details,
            'n_selected': len(sorted_selected),
            'correlation_threshold': correlation_threshold,
            'interaction_threshold': interaction_threshold,
            'shap_percentage': shap_percentage
        }

    except Exception as e:
        print(f"Erreur lors de la sélection des features: {str(e)}")
        import traceback
        print("Traceback complet:")
        print(traceback.format_exc())
        return None


def print_feature_selection_results(results):
    """
    Affiche les résultats de la sélection de features avec débogage
    """
    if not results:
        print("Pas de résultats à afficher")
        return

    print("\n=== DEBUG: Comparaison avec fichier source ===")
    print("Valeurs d'importance SHAP d'origine vs calculées:")

    for feature in results['selected_features'][:5]:  # Top 5 pour comparaison
        details = results['selected_features_with_details'][feature]
        print(f"\nFeature: {feature}")
        print(f"Valeur SHAP calculée: {details['shap_value']}")
        print(f"Importance absolue: {details['abs_shap_value']}")

    print(f"\nParamètres utilisés:")
    print(f"- Seuil de corrélation: {results['correlation_threshold']}")
    print(f"- Seuil d'interaction: {results['interaction_threshold']}")
    print(f"- Seuil SHAP cumulé visé: {results['shap_percentage'] * 100:.1f}%")

    print(f"\nAnalyse des corrélations et interactions:")
    n_correlations = sum(1 for details in results['removed_features_with_details'].values()
                         if 'correlated_with' in details)
    n_interactions = sum(1 for details in results['selected_features_with_details'].values()
                         if details.get('readded_via_interactions'))
    print(f"- Nombre de paires corrélées supprimées: {n_correlations}")
    print(f"- Nombre de features réintégrées par interactions: {n_interactions}")

    # Calculer l'importance totale et cumulative
    cumulative_importance_percentage = 0
    abs_shap_values = {f: abs(details['shap_value']) for f, details in results['selected_features_with_details'].items()}
    total_importance = sum(abs_shap_values.values())

    feature_rows = []

    for feature in results['selected_features']:
        details = results['selected_features_with_details'][feature]
        importance = details['shap_value']
        abs_importance = abs(importance)
        importance_percentage = (abs_importance / total_importance) * 100
        cumulative_importance_percentage += importance_percentage

        feature_rows.append({
            'feature': feature,
            'importance': importance,
            'importance_percentage': importance_percentage,
            'cumulative_importance_percentage': cumulative_importance_percentage
        })

    feature_df = pd.DataFrame(feature_rows)

    print(f"\nFeatures sélectionnées (par ordre d'importance SHAP): ({len(results['selected_features'])} features)")
    print(feature_df.to_string(index=False, formatters={
        'importance': '{:.8f}'.format,
        'importance_percentage': '{:.4f}'.format,
        'cumulative_importance_percentage': '{:.4f}'.format
    }))

    # Affichage des features supprimées en raison de corrélations élevées
    removed_due_to_correlation = []
    for feature, details in results['removed_features_with_details'].items():
        if 'correlated_with' in details:
            removed_due_to_correlation.append((feature, details))

    if removed_due_to_correlation:
        print(f"\nFeatures supprimées en raison de corrélations élevées (par ordre d'importance SHAP): ({len(removed_due_to_correlation)} features)")

        removed_corr_rows = []
        for feature, details in removed_due_to_correlation:
            importance = details['shap_value']
            abs_importance = abs(importance)
            correlated_with = details['correlated_with']
            kept_shap = details['kept_shap']
            correlation_value = details['correlation_value']
            removed_corr_rows.append({
                'feature_removed': feature,
                'shap_removed': importance,
                'feature_kept': correlated_with,
                'shap_kept': kept_shap,
                'correlation_value': correlation_value
            })

        removed_corr_df = pd.DataFrame(removed_corr_rows)
        removed_corr_df = removed_corr_df.sort_values('shap_removed', key=abs, ascending=False)

        print(removed_corr_df.to_string(index=False, formatters={
            'shap_removed': '{:.8f}'.format,
            'shap_kept': '{:.8f}'.format,
            'correlation_value': '{:.4f}'.format
        }))
    else:
        print("\nAucune feature supprimée en raison de corrélations élevées.")

    # Retourne les features au-dessus du seuil d'importance cumulée si nécessaire
    threshold_percent = results['shap_percentage'] * 100
    selected_features_threshold = feature_df[
        feature_df['cumulative_importance_percentage'] <= threshold_percent]['feature'].tolist()

    return selected_features_threshold


# Chemin d'accès et chargement des données
base_path = "C:/Users/aulac/OneDrive/Documents/Trading/PyCharmProject/MLStrategy/data_preprocessing/filtrageFeatures/4_0_6TP_1SL_18_12_301124"

correlation_df = load_data(os.path.join(base_path, "all_correlations.csv"))
shap_importance_df_train = load_data(os.path.join(base_path, "shap_dependencies_results", "shap_values_Training_Set.csv"))
print("\n=== DEBUG: Chargement des données ===")
print("Contenu du fichier SHAP train :")
print(shap_importance_df_train.head(10).to_string())
print("\nTypes des colonnes :")
print(shap_importance_df_train.dtypes)
# Vérification des valeurs spécifiques
print("\nVérification des features clés :")
for feature in ['bearish_volume_quality', 'bearish_absorption_score', 'bearish_market_context_score']:
    if feature in shap_importance_df_train['feature'].values:
        value = shap_importance_df_train[shap_importance_df_train['feature'] == feature]['importance'].iloc[0]
        print(f"{feature}: {value}")
    else:
        print(f"{feature}: Non trouvée")
shap_importance_df_test = load_data(os.path.join(base_path, "shap_dependencies_results", "shap_values_Test_Set.csv"))
interaction_df = load_data(os.path.join(base_path, "all_interactions_X_train.csv"))

# Utilisation de la fonction de sélection
results = select_features(
    correlation_df=correlation_df,
    shap_importance_df_train=shap_importance_df_train,
    shap_importance_df_test=shap_importance_df_test,
    interaction_df=interaction_df,
    top_n=len(shap_importance_df_train),
    correlation_threshold=0.90,
    interaction_threshold=15,
    shap_percentage=0.8
)

# Affichage des résultats avec récupération des features
selected_features_threshold = print_feature_selection_results(results)

# Affichage et enregistrement des features sélectionnées
print("\nAppuyez sur Entrée pour afficher et enregistrer la liste des features sélectionnées...")
input()

if selected_features_threshold:
    # Affichage dans la console
    print("\nselected_columnsByFiltering = [")
    for feature in selected_features_threshold:
        print(f"    '{feature}',")
    print("]")

    # Enregistrement dans un fichier
    output_file = os.path.join(base_path, "featuresfilteredByScript.txt")
    with open(output_file, 'w') as f:
        f.write("selected_columnsByFiltering = [\n")
        for feature in selected_features_threshold:
            f.write(f"    '{feature}',\n")
        f.write("]\n")

    print(f"\nListe des features enregistrée dans : {output_file}")
