import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from func_standard import timestamp_to_date_utc, print_notification, load_data
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1) PARAMÈTRES & CHARGEMENT
# ═════════════════════════════════════════════════════════════════════════════
import platform as platform_module
from path import Path
file_name="Step3_5_0_5TP_6SL_010124_010725_extractOnlyFullSession.csv"

#PATH_PROJECT="C:/Users/aurelienlachaud/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/"
dir_name = "5_0_5TP_6SL"

if platform_module.system() == "Darwin":  # macOS
    # Chemin vers Parallels Desktop VM Windows
    directory_path= Path(f"/Volumes/[C] Windows 11/Users/aurelienlachaud/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/{dir_name}/merge")
else:  # Windows
    directory_path= Path(f"C:/Users/aurelienlachaud/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/{dir_name}/merge")

file_path = directory_path / file_name  # Opérateur pathlib

user_choice = input("Entrée = shorts+longs, 's' = shorts only, 'l' = longs only : ").strip().lower()
df = load_data(file_path)

# ═════════════════════════════════════════════════════════════════════════════
# 2) PRÉ-TRAITEMENTS
# ═════════════════════════════════════════════════════════════════════════════
df['sc_timeStampOpening'] = pd.to_numeric(df['sc_timeStampOpening'], errors='coerce')
df['formatted_date']   = timestamp_to_date_utc(df['sc_timeStampOpening'])
df['date']             = pd.to_datetime(df['formatted_date'])
df['month']            = df['date'].dt.strftime('%Y-%m')

# ------------ construction de class_binaire (1 = succès, 0 = échec, 99 = ignore)
if user_choice == 's':                        # Shorts only
    mask = df['tradeDir'] == -1
    df['class_binaire'] = np.select(
        [(mask) & (df['tradeResult'] == 1),
         (mask) & (df['tradeResult'] == -1)],
        [1, 0], default=99)
    output_file_suffix = "_OnlyShort"

elif user_choice == 'l':                      # Longs only
    mask = df['tradeDir'] == 1
    df['class_binaire'] = np.select(
        [(mask) & (df['tradeResult'] == 1),
         (mask) & (df['tradeResult'] == -1)],
        [1, 0], default=99)
    output_file_suffix = "_OnlyLong"

else:                                         # Longs + Shorts
    df['class_binaire'] = np.select(
        [(df['tradeDir'] == 1)  & (df['tradeResult'] == 1),
         (df['tradeDir'] == -1) & (df['tradeResult'] == 1),
         (df['tradeDir'] == 1)  & (df['tradeResult'] == -1),
         (df['tradeDir'] == -1) & (df['tradeResult'] == -1)],
        [1, 1, 0, 0], default=99)
    output_file_suffix = ""

# ------------ Catégories texte pour le stackplot
df['trade_category'] = 'Pas de trade'
ok = df['class_binaire'] != 99
df.loc[ok, 'trade_category'] = np.select(
    [(df.loc[ok, 'tradeDir'] == 1)  & (df.loc[ok, 'tradeResult'] == 1),
     (df.loc[ok, 'tradeDir'] == 1)  & (df.loc[ok, 'tradeResult'] == -1),
     (df.loc[ok, 'tradeDir'] == -1) & (df.loc[ok, 'tradeResult'] == 1),
     (df.loc[ok, 'tradeDir'] == -1) & (df.loc[ok, 'tradeResult'] == -1)],
    ['Trades réussis long', 'Trades échoués long',
     'Trades réussis short', 'Trades échoués short'],
    default='Pas de trade')

all_categories = ['Trades réussis long', 'Trades réussis short',
                  'Trades échoués long', 'Trades échoués short']

# ═════════════════════════════════════════════════════════════════════════════
# 3) STATISTIQUES
# ═════════════════════════════════════════════════════════════════════════════
total_bougies   = len(df)
no_trade_cnt    = (df['tradeResult'] == 99).sum()
short_trade_cnt = (df['tradeDir'] == -1).sum()
long_trade_cnt  = (df['tradeDir'] ==  1).sum()

trades              = df[df['tradeResult'] != 99]
total_active_trades = len(trades)

short_trades = trades[trades['tradeDir'] == -1]
long_trades  = trades[trades['tradeDir'] ==  1]

short_fail, short_success = short_trades['class_binaire'].value_counts().reindex([0, 1]).fillna(0)
long_fail,  long_success  = long_trades ['class_binaire'].value_counts().reindex([0, 1]).fillna(0)

short_percent = 100 * len(short_trades) / total_active_trades if total_active_trades else 0
long_percent  = 100 * len(long_trades)  / total_active_trades if total_active_trades else 0

# Totaux globaux uniquement sur class_binaire ∈ {0,1}
active_mask    = trades['class_binaire'].isin([0, 1])
succ_tot       = int(trades.loc[active_mask, 'class_binaire'].sum())
fail_tot       = int(active_mask.sum() - succ_tot)      # 0 = échecs
total_active_trades = succ_tot + fail_tot               # cohérent

# ------------ Distributions mensuelles
monthly_distribution = (trades[active_mask]              # seulement 0/1
                        .groupby(['month', 'trade_category'])
                        .size().unstack(fill_value=0))
monthly_distribution['Total'] = monthly_distribution.sum(axis=1)
monthly_distribution = monthly_distribution.div(monthly_distribution['Total'], axis=0) * 100
for cat in all_categories:
    if cat not in monthly_distribution:
        monthly_distribution[cat] = 0

monthly_trade_counts = trades[active_mask].groupby('month').size()
monthly_cum_counts   = monthly_trade_counts.cumsum()
monthly_table = (pd.DataFrame({'Trades': monthly_trade_counts,
                               'Cumulé': monthly_cum_counts})
                 .reindex(monthly_distribution.index).fillna(0).astype(int))

# ═════════════════════════════════════════════════════════════════════════════
# 4) SAUVEGARDE CSV
# ═════════════════════════════════════════════════════════════════════════════
csv_out = os.path.join(os.path.dirname(file_path),
                       os.path.basename(file_path).replace(".csv", "").replace("Step3", "Step4")
                       + output_file_suffix + ".csv")
print_notification(f"Sauvegarde du DataFrame dans : {csv_out}")
df.to_csv(csv_out, sep=';', index=False, encoding='iso-8859-1')

# ═════════════════════════════════════════════════════════════════════════════
# 5) PRINT CONSOLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- INFORMATIONS ---\n")
print("1. Répartition des positions :")
print(f"Total bougies : {total_bougies}")
print(f"Aucun trade  : {no_trade_cnt} ({no_trade_cnt/total_bougies*100:.1f}%)")
print(f"Trades short : {short_trade_cnt} ({short_trade_cnt/total_bougies*100:.1f}%)")
print(f"Trades long  : {long_trade_cnt} ({long_trade_cnt/total_bougies*100:.1f}%)")

print("\n2. Trades actifs :")
print(f"Total actifs : {total_active_trades}")

if short_success + short_fail:
    pct_ok_short = short_success / (short_success + short_fail) * 100
    pct_ko_short = 100 - pct_ok_short
else:
    pct_ok_short = pct_ko_short = 0
print(f" Shorts ({short_percent:.1f}% du total actifs)")
print(f"   ↳ Réussis  : {short_success} ({pct_ok_short:.1f}%)")
print(f"   ↳ Échoués  : {short_fail}   ({pct_ko_short:.1f}%)")

if long_success + long_fail:
    pct_ok_long = long_success / (long_success + long_fail) * 100
    pct_ko_long = 100 - pct_ok_long
else:
    pct_ok_long = pct_ko_long = 0
print(f" Longs  ({long_percent:.1f}% du total actifs)")
print(f"   ↳ Réussis  : {long_success} ({pct_ok_long:.1f}%)")
print(f"   ↳ Échoués  : {long_fail}   ({pct_ko_long:.1f}%)")

print("\n3. Distribution mensuelle % :")
print(monthly_distribution[all_categories].round(1).to_string())

print("\n4. Global réussis / échoués :")
if total_active_trades:
    print(f"Trades réussis : {succ_tot} ({succ_tot/total_active_trades*100:.1f}%)")
    print(f"Trades échoués : {fail_tot} ({fail_tot/total_active_trades*100:.1f}%)")
else:
    print("Aucun trade actif.")
print("\n--- FIN ---\n")

# ═════════════════════════════════════════════════════════════════════════════
# 6) FIGURE (GridSpec 2 × 3)
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14))
gs  = gridspec.GridSpec(2, 3, height_ratios=[2.1, 2.0], hspace=0.35, wspace=0.28)

# Ligne 0
ax1 = fig.add_subplot(gs[0, 0])          # Camembert positions
ax2 = fig.add_subplot(gs[0, 1:3])        # Histogramme stacked

# Ligne 1
ax3 = fig.add_subplot(gs[1, 0])          # Stackplot mensuel
axT = fig.add_subplot(gs[1, 1])          # Tableau
ax4 = fig.add_subplot(gs[1, 2])          # Camembert global
axT.axis('off')

# (1) Camembert positions
ax1.pie([no_trade_cnt, short_trade_cnt, long_trade_cnt],
        labels=[f'Aucun\n{no_trade_cnt}', f'Shorts\n{short_trade_cnt}', f'Longs\n{long_trade_cnt}'],
        autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.05)
ax1.set_title('Répartition des positions')
ax1.axis('equal')

# (2) Histogramme réussite / échec
ax2.bar(['Short', 'Long'], [short_fail, long_fail],    color='red',   alpha=.7, label='Échoués')
ax2.bar(['Short', 'Long'], [short_success, long_success], bottom=[short_fail, long_fail],
        color='green', alpha=.7, label='Réussis')
ax2.set_ylabel('Nombre de trades')
ax2.set_title('Résultat des trades actifs')
ax2.legend()

# (3) Stackplot mensuel
colors = ['darkgreen', 'lightgreen', 'darkred', 'salmon']
ax3.stackplot(monthly_distribution.index,
              [monthly_distribution[c] for c in all_categories],
              colors=colors, labels=all_categories)
ax3.set_ylim(0, 100)
ax3.set_ylabel('%')
ax3.set_title('Distribution mensuelle détaillée')
ax3.set_xticks(range(len(monthly_distribution)))
ax3.set_xticklabels([pd.to_datetime(m).strftime('%b %y') for m in monthly_distribution.index],
                    rotation=90, fontsize=8)
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

# (4) Tableau
table = axT.table(cellText  = monthly_table.values,
                  rowLabels = [pd.to_datetime(idx).strftime('%b %y')
                               for idx in monthly_table.index],
                  colLabels = monthly_table.columns,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.35, 1.2)

# (5) Camembert global
if total_active_trades == 0:
    ax4.text(0.5, 0.5, "Aucun trade actif", ha='center', va='center', fontsize=12)
    ax4.axis('off')
else:
    ax4.pie([fail_tot, succ_tot],
            labels=[f'Échoués\n{fail_tot}', f'Réussis\n{succ_tot}'],
            autopct='%1.1f%%', colors=['red', 'green'],
            pctdistance=0.85, labeldistance=1.05)
    title = {'s': 'Résultats globaux (Shorts)',
             'l': 'Résultats globaux (Longs)'}.get(user_choice,
                                                  'Résultats globaux (Longs + Shorts)')
    ax4.set_title(title)
    ax4.axis('equal')

plt.show()
