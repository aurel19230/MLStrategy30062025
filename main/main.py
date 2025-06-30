import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# 1️⃣  Renseigne ici les cinq chiffres qui figurent dans la légende de chaque subplot
# ─── Exemple de structure — remplace par tes vraies valeurs ───
data = [
    # nom,                         P25,   P50,   P75
    ("test_Toutes",                1.638, 1.706, 1.728),
    ("val1_Toutes",                1.649, 1.710, 1.758),
    ("val_Toutes",                 1.649, 1.762, 1.810),
    ("test_[0_1]",                 1.578, 1.676, 1.770),
    ("val1_[0_1]",                 1.615, 1.691, 1.794),
    ("val_[0_1]",                  1.678, 1.760, 1.830),
    ("test_[2_6]",                 1.643, 1.706, 1.735),
    ("val1_[2_6]",                 1.649, 1.703, 1.752),
    ("val_[2_6]",                  1.727, 1.756, 1.799),
]

df = pd.DataFrame(data, columns=["dist", "P25", "P50", "P75"])
df["IQR"] = df["P75"] - df["P25"]
df["Bowley"] = (df["P75"] + df["P25"] - 2*df["P50"]) / df["IQR"]

# 2️⃣  Standardisation sur les 5 composantes
X = df[["P25", "P50", "P75", "IQR", "Bowley"]].values
X_scaled = StandardScaler().fit_transform(X)

# 3️⃣  Distances euclidiennes
dist = squareform(pdist(X_scaled, "euclidean"))
dist_df = pd.DataFrame(dist, index=df["dist"], columns=df["dist"])

display(dist_df)
