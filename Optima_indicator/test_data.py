import pandas as pd

path1 = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\\5_0_5TP_6SL_samde_soir\merge\Step4_5_0_5TP_6SL_010124_110625_extractOnlyFullSession_OnlyShort.csv"

# Lecture avec le bon encodage et séparateur
df = pd.read_csv(
    path1,
    sep=';',
    encoding='ISO-8859-1',    # 🔥 Résout le UnicodeDecodeError
    parse_dates=True,
    dayfirst=True
)
print(df['sc_deltaTimestampOpening'].head(10))

# Filtrer pour ne garder que les lignes avec class_binaire == 0 ou 1
df_filtered = df[df['class_binaire'].isin([0, 1])]

# Afficher les colonnes demandées
print(df_filtered['sc_deltaTimestampOpening'].head(10))
print(df_filtered['sc_volPocVolRevesalXContRatio'].head(10))

print(df_filtered['volcontZone_zoneReversal'].head(10))
print(df_filtered['sc_volcontZone_zoneReversal'].head(10))
print(df_filtered['deltaRev_volRev_ratio'].head(10))
print(df_filtered['sc_deltaRev_volRev_ratio'].head(10))
print(df_filtered['volRevVolRevesalXContRatio'].head(10))
print(df_filtered['sc_volRevVolRevesalXContRatio'].head(10))


print(df_filtered['sc_deltaAbv'].head(10))
print(df_filtered['sc_deltaBlw'].head(10))
print(df_filtered['sc_volAbv'].head(10))
print(df_filtered['sc_volBlw'].head(10))


