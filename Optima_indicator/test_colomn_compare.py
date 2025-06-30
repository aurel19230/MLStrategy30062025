import pandas as pd

# Chemins des fichiers CSV
path1 = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL\merge\Step5_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_OnlyLong_feat__split3_30092024_27022025.csv"
path2 = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL_lundi_soir\merge\Step5_5_0_5TP_6SL_010124_090625_extractOnlyFullSession_OnlyShort_feat__split3_30092024_27022025.csv"

# Chargement avec encodage ISO-8859-1 et séparateur ";"
df1 = pd.read_csv(path1, sep=';', encoding='ISO-8859-1')
df2 = pd.read_csv(path2, sep=';', encoding='ISO-8859-1')

# Liste des colonnes que tu veux comparer (adapter ici)
cols_to_compare = [
     "sc_timeStampOpening",
    #  "class_binaire",
   # "sc_high"
    # 🔁 Ajoute ici d'autres colonnes à comparer
]

# Vérification : ne garder que les colonnes présentes dans les deux fichiers
common_cols = [col for col in cols_to_compare if col in df1.columns and col in df2.columns]

# Réduction aux colonnes spécifiées
df1_common = df1[common_cols].copy()
df2_common = df2[common_cols].copy()

# Tronque à la même longueur si nécessaire
min_len = min(len(df1_common), len(df2_common))
df1_common = df1_common.iloc[:min_len]
df2_common = df2_common.iloc[:min_len]

# Comparaison ligne par ligne
differences = []
for i in range(min_len):
    row1 = df1_common.iloc[i]
    row2 = df2_common.iloc[i]
    diff = row1 != row2
    if diff.any():
        diff_cols = row1.index[diff].tolist()
        differences.append({
            "index": i,
            "date1": row1.get("date", None),
            "date2": row2.get("date", None),
            "colonnes_différentes": diff_cols,
            "valeurs_fichier1": row1[diff].to_dict(),
            "valeurs_fichier2": row2[diff].to_dict()
        })
    if len(differences) >= 100000:
        break
print(len(differences))
# Affichage des différences
for diff in differences:
    date_info = diff.get("date1", "❓")
    print(f"\n🔸 Différence à l’index {diff['index']} | 📅 date = {date_info}")
    print(f"   Colonnes différentes : {diff['colonnes_différentes']}")
    print(f"   Valeurs fichier 1 : {diff['valeurs_fichier1']}")
    print(f"   Valeurs fichier 2 : {diff['valeurs_fichier2']}")
    print(len(differences))