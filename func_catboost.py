import numpy as np

class WeightedLogisticObjective:
    def __init__(self, w_p, w_n):
        self.w_p = w_p
        self.w_n = w_n

    def calc_ders_range(self, approxes, targets, weights):
        # approxes est une liste, on prend le premier élément pour un problème binaire
        approx = approxes[0]
        targets = np.array(targets)

        # Calcul du sigmoid
        predt_sigmoid = 1 / (1 + np.exp(-approx))

        # Calcul des gradients et hessiens
        grad = predt_sigmoid - targets
        hess = predt_sigmoid * (1 - predt_sigmoid)

        # Application des poids de classe
        weights = np.where(targets == 1, self.w_p, self.w_n)
        grad *= weights
        hess *= weights

        # Retourner les gradients et hessiens sous forme de liste de tuples
        return list(zip(grad, hess))

class ProfitBasedMetric:
    def __init__(self, metric_dict, normalize=False):
        self.metric_dict = metric_dict
        self.normalize = normalize

    def get_final_error(self, approxes, targets, weights):
        # Récupération des prédictions (binaire)
        approx = approxes[0]
        targets = np.array(targets)

        # Application du sigmoid
        predt_sigmoid = 1 / (1 + np.exp(-approx))

        # Application du seuil
        threshold = self.metric_dict.get('threshold', 0.55555555)
        y_pred_threshold = (predt_sigmoid > threshold).astype(int)

        # Calcul des métriques
        total_profit, tp, fp = self.calculate_profit(targets, y_pred_threshold)

        # Normalisation
        if self.normalize:
            total_trades = tp + fp
            final_profit = total_profit / total_trades if total_trades > 0 else total_profit
            metric_name = 'custom_metric_ProfitBased_norm'
        else:
            final_profit = total_profit
            metric_name = 'custom_metric_ProfitBased'

        # Retour : (nom_metrique, valeur, is_bigger_better)
        return metric_name, float(final_profit), True

    def calculate_profit(self, y_true, y_pred_threshold):
        tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
        fp = np.sum((y_true == 0) & (y_pred_threshold == 1))
        fn = np.sum((y_true == 1) & (y_pred_threshold == 0))

        profit_per_tp = self.metric_dict.get('profit_per_tp', 1.0)
        loss_per_fp = self.metric_dict.get('loss_per_fp', -1.1)
        penalty_per_fn = self.metric_dict.get('penalty_per_fn', -0.1)

        total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)
        return float(total_profit), int(tp), int(fp)
