from sklearn.model_selection import BaseCrossValidator
import numpy as np


class nonAnchore_dWalkForwardCV_afterPrevTrain(BaseCrossValidator):
    """
    Validateur walk-forward 'non ancré' (nonAnchored) qui calcule
    automatiquement la taille de l'entrainement (train_size).

    - Nombre de splits (folds) : n_splits
    - Ratio pour la validation : val_ratio (<= 1)

    Caractéristiques :
    ----------------
    1. Le train du fold suivant démarre là où s'arrête le train du fold précédent
       (pas de recouvrement du train).
    2. La validation est immédiatement après le train.
    3. train_size est calculé à partir de la formule :
         floor(n / ((1 + val_ratio) * n_splits))
    4. val_size = round(train_size * val_ratio).
    5. Si on ne peut pas générer l'ensemble des n_splits demandés
       parce qu'on atteint la fin du dataset, on arrête.
    """

    def __init__(self, n_splits, val_ratio=0.5):
        super().__init__()
        assert 0 < val_ratio <= 1, "val_ratio doit être compris entre 0 et 1."

        self.n_splits = n_splits
        self.val_ratio = val_ratio

        # train_size sera calculé dans .split() en fonction de la taille de X

    def get_n_splits(self, X, y=None, groups=None):
        """
        On retourne simplement le nombre de splits souhaité,
        même si en pratique on risque d'en produire moins
        si le dataset est trop petit.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)

        # Pour n points et f folds avec ratio r:
        # f * (train_size + val_size) = n
        # f * (train_size + train_size * val_ratio) = n
        # f * train_size * (1 + val_ratio) = n
        train_size = int(n / (self.n_splits + self.val_ratio))
        val_size = int(train_size * self.val_ratio)

        # Ajuster pour l'espace restant
        total_used = self.n_splits * (train_size + val_size)
        leftover = n - total_used
        if leftover > 0:
            # Augmenter le dernier train_size pour utiliser l'espace restant
            last_train_extra = leftover

        start = 0
        for fold in range(self.n_splits):
            current_train_size = train_size
            """
            if fold == self.n_splits - 1:  # dernier fold
                current_train_size += last_train_extra
            """
            end_train = start + current_train_size
            end_val = end_train + val_size

            if end_val > n:  # Sécurité
                break

            yield (range(start, end_train), range(end_train, end_val))
            start += current_train_size


# ------------------- EXEMPLE D’UTILISATION -------------------
if __name__ == "__main__":
    import pandas as pd

    data = pd.Series(range(297787))  # 400 points

    # Suppose qu'on veut 3 splits et un val_ratio=0.5
    # => on ne fixe PAS train_size, il sera calculé
    n_splits = 5
    r = 0.7

    cv = nonAnchore_dWalkForwardCV_afterPrevTrain(
        n_splits=n_splits,
        val_ratio=r
    )

    print("Nombre de splits (demande) =", cv.get_n_splits(data))

    for i, (train_idx, val_idx) in enumerate(cv.split(data), start=1):
        print(f"\n--- Fold {i} ---")
        print(f"Train : {train_idx.start} -> {train_idx.stop} (taille = {len(train_idx)})")
        print(f"Valid : {val_idx.start} -> {val_idx.stop}   (taille = {len(val_idx)})")
