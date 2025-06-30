import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from standardFunc_sauv import timestamp_to_date_utc

# Charger les données
file_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_4TP_1SL\\merge\\test\\010424_300824_4TicksRevdepuisavril.csv"
df = pd.read_csv(file_path, delimiter=';')

# Calcul correct de la variable cible
df['tradeOutcome'] = np.where(df['tradeResult'] == 99, 99,
                              np.where(df['tradeResult'] == 1, 1, 0))
df['trade_result4Optim'] = np.where(df['tradeOutcome'] == 99, 99,
                                    df['tradeDir'] * df['tradeOutcome'])

# Création des classes selon les instructions
df['class'] = df['trade_result4Optim']

# Mapper les valeurs numériques aux descriptions des classes
class_mapping = {
    -1: 'Short réussi',
    0: 'Trade échoué',
    1: 'Long réussi',
    99: 'Pas de trade'
}
df['class_description'] = df['class'].map(class_mapping)

# Convertir la colonne timeStamp en datetime
df['formatted_date'] = timestamp_to_date_utc(df['timeStampOpening'])
df['date'] = pd.to_datetime(df['formatted_date'])
df['month'] = df['date'].dt.to_period('M')

# Filtrer les données pour exclure les cas où aucun trade n'est pris
df_trades = df[df['class'] != 99].copy()

# Calculer le poids de chaque classe (excluant les 'Pas de trade')
class_weights = df_trades['class_description'].value_counts(normalize=True)

print("Poids de chaque classe (excluant 'Pas de trade'):")
print(class_weights)

# Visualiser la distribution des classes
plt.figure(figsize=(12, 6))
sns.countplot(x='class_description', data=df_trades, order=class_weights.index)
plt.title('Distribution des classes (excluant Pas de trade)')
plt.xlabel('Classe')
plt.ylabel('Nombre de trades')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculer des statistiques supplémentaires
total_trades = len(df_trades)
successful_trades = len(df_trades[df_trades['class'].isin([-1, 1])])
win_rate = successful_trades / total_trades

print(f"\nNombre total de trades : {total_trades}")
print(f"Nombre de trades réussis : {successful_trades}")
print(f"Win rate global : {win_rate:.2%}")

# Analyse séparée pour les trades longs et courts
long_trades = df_trades[df_trades['tradeDir'] == 1]
short_trades = df_trades[df_trades['tradeDir'] == -1]

print("\nAnalyse des trades longs:")
print(f"Nombre de trades longs : {len(long_trades)}")
print(f"Win rate des trades longs : {(long_trades['class'] == 1).mean():.2%}")

print("\nAnalyse des trades courts:")
print(f"Nombre de trades courts : {len(short_trades)}")
print(f"Win rate des trades courts : {(short_trades['class'] == -1).mean():.2%}")

# Calculer et afficher les ratios de déséquilibre
max_class = class_weights.idxmax()
min_class = class_weights.idxmin()
imbalance_ratio = class_weights[max_class] / class_weights[min_class]

print(f"\nRatio de déséquilibre (classe majoritaire / classe minoritaire): {imbalance_ratio:.2f}")
print(f"Classe majoritaire: {max_class} ({class_weights[max_class]:.2%})")
print(f"Classe minoritaire: {min_class} ({class_weights[min_class]:.2%})")

# Visualiser la distribution des classes par mois
monthly_class_dist = df_trades.groupby(['month', 'class_description']).size().unstack(fill_value=0)
monthly_class_dist_norm = monthly_class_dist.div(monthly_class_dist.sum(axis=1), axis=0)

plt.figure(figsize=(15, 8))
monthly_class_dist_norm.plot(kind='area', stacked=True)
plt.title('Distribution mensuelle des classes (excluant Pas de trade)')
plt.xlabel('Mois')
plt.ylabel('Proportion')
plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Sauvegarder le DataFrame complet dans un fichier CSV
output_file_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_4TP_1SL\\merge\\df_complet_analyse.csv"
df.to_csv(output_file_path, index=False, sep=';')
print(f"\nLe DataFrame complet a été sauvegardé dans : {output_file_path}")