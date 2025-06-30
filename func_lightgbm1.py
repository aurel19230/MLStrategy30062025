import lightgbm as lgb

from definition import *
from sklearn.metrics import brier_score_loss


# Fonction objective personnalisée pour LightGBM
def lgb_weighted_logistic_objective(w_p: float, w_n: float):
    def weighted_logistic_obj(preds, train_data):
        y_true = train_data.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Application de la sigmoid

        # Calcul du gradient et de la hessienne avec les poids
        weights = np.where(y_true == 1, w_p, w_n)
        grad = (preds - y_true) * weights
        hess = preds * (1.0 - preds) * weights

        return grad, hess

    return weighted_logistic_obj


# Fonction auxiliaire pour le calcul du profit (équivalent à xgb_calculate_profitBased_gpu)
"""
def lgb_calculate_profitBased(y_true=None, y_pred_threshold=None, metric_dict=None,config=None):
    """
   # Calcule les métriques de profit pour LightGBM
"""
    # Calcul des métriques de base
    tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_true == 0) & (y_pred_threshold == 1))
    fn = np.sum((y_true == 1) & (y_pred_threshold == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = config.get('profit_per_tp', 11111)
    loss_per_fp = config.get('loss_per_fp', 11111)
    penalty_per_fn = metric_dict.get('penalty_per_fn', 11111)
    if (penalty_per_fn == 11111 or profit_per_tp == 11111 or loss_per_fp == 11111):
        raise ValueError(
            f"Paramètres invalides : penalty_per_fn={penalty_per_fn}, profit_per_tp={profit_per_tp}, loss_per_fp={loss_per_fp}")

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)
"""


def lgb_custom_metric_pnl_factory(
        config=None,
        other_params=None,
):
    # Créez la liste d'historique en dehors de la fonction interne
    tp_fp_history = []

    def custom_metric_pnl(preds, train_data):
        """
        Métrique LightGBM pour le profit,
        utilisant config, metric_dict, etc.
        """
        # Récupération du vecteur de labels
        y_true = train_data.get_label()

        # Récupération de la série PnL attachée au dataset
        y_pnl_data = train_data.pnl_data  # On suppose que vous avez fait ltrain.pnl_data = y_pnl_data_train_cv

        # Application de la sigmoid
        preds_sigmoid = 1.0 / (1.0 + np.exp(-preds))
        preds_sigmoid = np.clip(preds_sigmoid, 0.0, 1.0)

        # Récupération d’un paramètre (ex. threshold) dans metric_dict
        threshold = other_params.get('threshold', 0.55555555)
        y_pred_threshold = (preds_sigmoid > threshold).astype(np.int32)


        # Pas besoin de y_pnl_data_train_cv ni y_pnl_data_val_cv_OrTest
        # dans ce mode "direct", on suppose que c'est déjà le bon vecteur

        if y_pnl_data is None:
            raise ValueError("Il faut fournir y_pnl_data_array pour calculer le profit.")

        y_true_class_binaire = np.asarray(y_true)
        y_pred = np.asarray(y_pred_threshold)
        y_pnl_data_array = np.asarray(y_pnl_data)

        # Vérifier l'alignement
        if not (len(y_true_class_binaire) == len(y_pred) == len(y_pnl_data_array)):
            raise ValueError(f"Mauvais alignement: {len(y_true_class_binaire)=}, {len(y_pred)=}, {len(y_pnl_data_array)=}")

        # Calcul masques
        tp_mask = (y_true_class_binaire == 1) & (y_pred == 1)
        fp_mask = (y_true_class_binaire == 0) & (y_pred == 1)
        fn_mask = (y_true_class_binaire == 1) & (y_pred == 0)

        # Comptage
        tp = np.sum(tp_mask)
        fp = np.sum(fp_mask)
        fn = np.sum(fn_mask)

        # Récupération des pénalités dans metric_dict/config
        penalty_per_fn = other_params.get('penalty_per_fn', config.get('penalty_per_fn', 0))

        # Calcul
        tp_profits = np.sum(np.maximum(0, y_pnl_data_array[tp_mask])) if tp > 0 else 0
        fp_losses = np.sum(np.minimum(0, y_pnl_data_array[fp_mask])) if fp > 0 else 0
        fn_penalty = fn * penalty_per_fn

        total_profit = tp_profits + fp_losses + fn_penalty
        # Récupération des PnL utilisés pour les TP et FP
        tp_pnls = y_pnl_data_array[tp_mask]
        fp_pnls = y_pnl_data_array[fp_mask]

        # Vérification que toutes les valeurs de TP sont égales à 175
        tp_unique_values = np.unique(tp_pnls)
        if not np.all(tp_unique_values == 175):
            print(f"❌ Incohérence : TP contient des valeurs autres que 175 : {tp_unique_values}")
            exit(100)
        # else:
        #     print("✅ Tous les TP ont une valeur de 175.")
        #     exit(101)


        # Vérification que toutes les valeurs de FP sont égales à -227
        fp_unique_values = np.unique(fp_pnls)
        if not np.all(fp_unique_values == -227):
            print(f"❌ Incohérence : FP contient des valeurs autres que -227 : {fp_unique_values}")
            exit(102)

        # else:
        #     print("✅ Tous les FP ont une valeur de -227.")

        #print(preds)
        total_profit = (tp * 175) + (fp * -227)

        # Stockez simplement les valeurs TP et FP pour chaque itération
        # Sans condition sur evals_result qui n'est pas disponible ici
        tp_fp_history.append((tp, fp))

        return 'custom_metric_PNL', float(total_profit), True

        # Attachez l'historique à la fonction

    custom_metric_pnl.tp_fp_history = tp_fp_history
    return custom_metric_pnl


def train_and_evaluate_lightgbm_model(
        X_train_cv=None,
        X_val_cv=None,
        y_train_cv=None,
        y_val_cv=None,
        y_pnl_data_train_cv=None,
        y_pnl_data_val_cv_OrTest=None,
        params_optuna=None,
        other_params=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=0,
):
    ltrain = lgb.Dataset(X_train_cv, label=y_train_cv)
    # on attache y_pnl_data_train_cv dans ltrain
    ltrain.pnl_data = y_pnl_data_train_cv

    lval = lgb.Dataset(X_val_cv, label=y_val_cv)
    # on attache y_pnl_data_val_cv_OrTest dans lval
    lval.pnl_data = y_pnl_data_val_cv_OrTest

    # Configuration des paramètres selon l'objectif
    custom_objective_lossFct = config.get('custom_objective_lossFct', 13)

    params_optuna.update({
        'objective': lgb_weighted_logistic_objective(other_params['w_p'], other_params['w_n']),
        'metric': None,
        'device_type': 'cpu'
    })

    custom_metric_function = lgb_custom_metric_pnl_factory(config=config, other_params=other_params)

    params_optuna['early_stopping_rounds'] = config.get('early_stopping_rounds', 13)
    params_optuna['verbose'] = -1

    print(f"Boosting type configuré : {params_optuna['boosting_type']}")
    print(f"num_boost_round : {other_params['num_boost_round']}")

    evals_result = {}
    # 3) Appeler l'entraînement
    current_model = lgb.train(
        params=params_optuna,
        train_set=ltrain,
        num_boost_round=other_params['num_boost_round'],
        valid_sets=[ltrain, lval],
        valid_names=['train', 'eval'],
        feval=custom_metric_function,
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=log_evaluation)  # Affiche les logs toutes les 10 itérations
        ],
    )

    eval_scores = evals_result['eval']['custom_metric_PNL']
    train_scores = evals_result['train']['custom_metric_PNL']
    val_best = max(eval_scores)
    best_idx = eval_scores.index(val_best)
    best_iteration = best_idx + 1
    train_best = train_scores[best_idx]

    # SOLUTION RADICALE : Recalculer les TP et FP exactement comme dans custom_metric_pnl
    # Récupérer les prédictions brutes à l'itération optimale
    best_preds_raw = current_model.predict(X_val_cv, num_iteration=best_idx+1, raw_score=True)

    # Appliquer la sigmoid exactement comme dans custom_metric_pnl
    preds_sigmoid = 1.0 / (1.0 + np.exp(-best_preds_raw))
    preds_sigmoid = np.clip(preds_sigmoid, 0.0, 1.0)

    # Appliquer le seuil
    threshold = other_params.get('threshold', 0.55555555)  # Même valeur que dans custom_metric_pnl
    predictions = (preds_sigmoid > threshold).astype(np.int32)

    # Calculer TP/FP exactement comme dans la fonction de métrique
    tp_recalculated = np.sum((y_val_cv == 1) & (predictions == 1))
    fp_recalculated = np.sum((y_val_cv == 0) & (predictions == 1))
    pnl_recalculated = (tp_recalculated * 175) + (fp_recalculated * -227)

    print(f"TP/FP recalculés: {tp_recalculated}/{fp_recalculated}")
    print(f"PnL recalculé: {pnl_recalculated}")

    # Vérification
    if val_best == pnl_recalculated:
        print("✅ val_best est bien égal au PnL recalculé.")
    else:
        print(f"❌ Mismatch : val_best ({val_best}) est différent du PnL recalculé ({pnl_recalculated}).")
        print(f"   Différence: {val_best - pnl_recalculated}")

    # Utiliser ces valeurs recalculées pour la suite
    tp_val = tp_recalculated
    fp_val = fp_recalculated

    # Pour garder la cohérence avec le reste du code
    model_type = config['model_type']
    val_pred_proba_log_odds = best_preds_raw
    pred_proba_afterSig = preds_sigmoid

    # Calcul de la matrice de confusion complète
    if config['device_'] != 'cpu':
        import cupy as cp
        y_true_converted = cp.asarray(y_val_cv)
        predictions_converted = cp.asarray(predictions)
    else:
        y_true_converted = y_val_cv
        predictions_converted = predictions

        # Calcul de la matrice de confusion (fn_val et tn_val sont nécessaires ailleurs)
        tn_val, fp_val, fn_val, tp_val = compute_confusion_matrix_cpu(y_true_converted, predictions_converted, config)

    if (current_model.best_iteration != best_idx ):
        print(
            f"Différence détectée: current_model.best_iteration={current_model.best_iteration}, best_idx+1={best_idx + 1}")
        exit()


    # Garder la cohérence en utilisant best_idx au lieu de best_iteration pour train aussi
    y_train_predProba, train_pred_proba_log_odds, train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = \
        predict_and_compute_metrics_XgbOrLightGbm(model=current_model, X_data=X_train_cv, y_true=y_train_cv,
                                                  best_iteration=best_idx+1, threshold=other_params['threshold'],
                                                  config=config)

    # Calcul du PnL attendu - maintenant il devrait correspondre à val_best
    pnl_calcule = (tp_val * 175) + (fp_val * -227)

    print("train_and_evaluate_lightgbm_model", other_params['threshold'])
    print(f"best_idx={best_idx} val_best : {val_best} ")
    print(f"PnL calculé final : {pnl_calcule} | Vérif : TP = {tp_val}, FP = {fp_val}")

    # Le reste du code reste inchangé...

    # Vérification
    if val_best == pnl_calcule:
        print("✅ val_best est bien égal au PnL calculé.")
    else:
        print("❌ Mismatch : val_best est différent du PnL calculé.")

    # 📉 Brier Score brut
    brier = brier_score_loss(y_val_cv, pred_proba_afterSig)
    # print("brier score: ",brier)
    # 📊 Brier Score baseline
    baseline_pred = np.full_like(y_val_cv, y_val_cv.mean(), dtype=np.float64)
    baseline_brier = brier_score_loss(y_val_cv, baseline_pred)
    relative_brier = brier / baseline_brier if baseline_brier > 0 else brier

    # Construction des métriques
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': val_best,
        'best_iteration': best_iteration,
        'brier':brier,
        'relative_brier':relative_brier,
        'y_val_cv':y_val_cv,
        'val_pred_proba_log_odds':val_pred_proba_log_odds,
    }
    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(Y_train_cv),
        'train_bestVal_custom_metric_pnl': train_best
    }
    #print(
    #   f"Val metrics: tp={val_metrics['tp']}, fp={val_metrics['fp']}, tn={val_metrics['tn']}, fn={val_metrics['fn']}, samples={val_metrics['total_samples']}, best_pnl={val_metrics['val_bestVal_custom_metric_pnl']:.4f}, best_iter={val_metrics['best_iteration']}")
    #print(
    #   f"Train metrics: tp={train_metrics['tp']}, fp={train_metrics['fp']}, tn={train_metrics['tn']}, fn={train_metrics['fn']}, samples={train_metrics['total_samples']}, best_pnl={train_metrics['train_bestVal_custom_metric_pnl']:.4f}")
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100,
                                       2) if tp_fp_tn_fn_sum_train else 0.00
    val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00
    perctDiff_ratioTradeSample_train_val = abs(
        calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config))
    winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))

    # Compilation des statistiques du fold
    fold_stats = compile_fold_stats(
        fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_best, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )
    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    debug_info = compile_debug_info(other_params, config, pred_proba_afterSig, y_train_predProba)

    return {
        'current_model': current_model,
        'fold_raw_data': fold_raw_data,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': evals_result,
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }

