"""
Fusionne les fichiers Step1_*_X.csv placés dans le dossier ...\merge
  • Concatène dans l’ordre _0, _1, _2 …         (obligatoire & consécutif)
  • Option ‘d’ : dé-doublonne selon un groupe de colonnes et
                 conserve le plus petit timeStampOpening
  • Vérifie qu’à l’issue la colonne timeStampOpening est
    STRICTEMENT croissante ; sinon, affiche les paires fautives
    puis lève ValueError.

Sortie : Step2_<config>_<startDate>_<endDate>.csv (séparateur ‘;’)
------------------------------------------------------------------------
© 2025 – script destiné à un usage interne
"""

# ───────────────────────────────── Imports ────────────────────────────────────
import os
import re
import numpy as np
import pandas as pd

# ───────────────────────── Paramétrage dossier ───────────────────────────────
PATH_PROJECT="C:/Users/aurelienlachaud/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/"
DIR = "5_0_5TP_6SL"


import platform as platform_module
from pathlib import Path  # ✅ pathlib au lieu de path
import os

# Construction du chemin
if platform_module.system() != "Darwin":
    directory = Path(rf"{PATH_PROJECT}{DIR}/merge")

print(directory)

# ✅ Utiliser pathlib.parts au lieu de split
path_components = directory.parts
print(f"Path components: {path_components}")

if "merge" in path_components:
    merge_index = path_components.index("merge")
    xtickRev_config_dir = path_components[merge_index - 1]
    print(f"Config directory: {xtickRev_config_dir}")
else:
    print("'merge' not found in path components")

# ─────────────────────────── Choice de l’utilisateur ─────────────────────────
option = input(
    "Appuyez sur 'd' pour dé-doublonner, Entrée pour concaténer simplement : "
).lower()

# ───────────────────── Function : anomalies chronologiques ───────────────────
def show_timestamp_anomalies(timestamps: np.ndarray, label: str = "sc_timeStampOpening"):
    """Affiche les inversions strictes t[i] > t[i+1] et lève ValueError."""
    bad_idx = np.where(np.diff(timestamps) < 0)[0]
    if bad_idx.size == 0:
        print(f"✔️  Aucun problème d’ordre chronologique détecté dans « {label} » "
              f"({len(timestamps)} valeurs).")
        return
    print(f"\n❌  {bad_idx.size} inversion(s) détectée(s) dans « {label} » :")
    for i in bad_idx:
        print(f"  • ligne {i+1:>8} : {int(timestamps[i])}  →  ligne {i+2:>8} : "
              f"{int(timestamps[i+1])}")
    print("\nArrêt du traitement car l’ordre n’est pas strictement croissant.")
    raise ValueError("Impossible d’obtenir un ordre chronologique strict des timeStampOpening.")

# ───────────────────── Function : nom de fichier de sortie ───────────────────
def generate_output_filename(files, config_dir):
    if not files:
        raise ValueError("La liste des fichiers est vide.")
    sorted_files = sorted(files)
    first_file   = next((f for f in sorted_files if f.endswith("_0.csv")), None)
    if first_file is None:
        raise ValueError("Aucun fichier ne se termine par '_0.csv'.")
    start_date   = first_file.split("_")[1]
    last_file    = max(sorted_files,
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))
    end_date     = last_file.split("_")[2]
    return f"Step2_{config_dir}_{start_date}_{end_date}.csv"

# ─────────────────────────── Function : merge files ──────────────────────────
def merge_files(directory_path: str) -> pd.DataFrame:
    # Liste et tri des fichiers *_X.csv
    all_csv   = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    files     = [f for f in all_csv if re.match(r".+_\d+\.csv$", f)]
    files.sort(key=lambda f: int(re.findall(r"_(\d+)\.csv$", f)[0]))

    print("Fichiers à fusionner :")
    for f in files:
        print(f" - {f}")
    print()
    if not files:
        raise ValueError("Aucun fichier *_X.csv trouvé dans le dossier.")

    file_numbers = [int(re.findall(r"_(\d+)\.csv$", f)[0]) for f in files]
    if file_numbers != list(range(len(file_numbers))):
        raise ValueError("Les fichiers ne sont pas consécutifs ou ne commencent pas par *_0.csv.")

    # Lecture séquentielle
    dfs = []
    for idx, file in enumerate(files):
        fp = os.path.join(directory_path, file)
        df = pd.read_csv(fp, delimiter=";", header=0)
        print(f"Traitement du fichier {file} : {len(df)} lignes")
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # ------------------- Option de dé-doublonnage ----------------------------
    if option == "d":
        print("\n🧹  Dé-doublonnage activé")
        merged_df["sc_timeStampOpening"] = pd.to_numeric(merged_df["sc_timeStampOpening"])
        cols_check = [
            "close", "open", "high", "low", "volume",
            "atr", "vaDelta_6periods", "vaVol_16periods"
        ]
        # Analyse pré-suppression
        duplicates = merged_df[merged_df.duplicated(subset=cols_check, keep=False)].copy()
        if not duplicates.empty:
            for keys, grp in duplicates.groupby(cols_check):
                n_del = len(grp) - 1
                keep_ts = grp["sc_timeStampOpening"].min()
                print("\nGroupe de doublons:")
                print(f"Nombre de lignes à supprimer : {n_del}")
                print(f"sc_timeStampOpening conservé    : {keep_ts}")
                print(grp.sort_values("sc_timeStampOpening"))
                print("-" * 80)

        merged_df = (merged_df
                     .sort_values("sc_timeStampOpening")
                     .drop_duplicates(subset=cols_check, keep="first")
                     .reset_index(drop=True))

        print("\nDé-doublonnage terminé.")
        print(f"Lignes après nettoyage : {len(merged_df)}")

    # -------------------- Vérification chronologique -------------------------
    if not merged_df["sc_timeStampOpening"].is_monotonic_increasing:
        show_timestamp_anomalies(
            merged_df["sc_timeStampOpening"].to_numpy(dtype=np.int64),
            label="sc_timeStampOpening"
        )  # lève déjà ValueError
    else:
        print("Les timeStampOpening sont strictement croissants.")

    print(f"\nNombre total de lignes après fusion : {len(merged_df)}")
    return merged_df

# ───────────────────────── Exécution principale ──────────────────────────────
try:
    merged = merge_files(directory)

    files_for_name = [f for f in os.listdir(directory) if re.match(r".+_\d+\.csv$", f)]
    output_name    = generate_output_filename(files_for_name, xtickRev_config_dir)
    output_path    = os.path.join(directory, output_name)

    merged.to_csv(output_path, index=False, sep=";")
    print(f"\n✅  Fusion terminée. Résultat sauvegardé : {output_path}")

except ValueError as err:
    print(f"\nErreur : {err}")

except Exception as exc:
    print(f"\nUne erreur inattendue s’est produite : {exc}")
