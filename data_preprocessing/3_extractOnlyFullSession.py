import pandas as pd
import numpy as np
from numba import jit
import os
import time
from func_standard import  print_notification

import platform as platform_module
from path import Path

# ────────────────────────────────────────────────────────────────────────────────
# Construction du chemin de base selon l'OS
# ────────────────────────────────────────────────────────────────────────────────
PATH_PROJECT="C:/Users/aurelienlachaud/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/"
DIR = "5_0_5TP_6SL"
if platform_module.system() != "Darwin":
    directory = Path(
        rf"{PATH_PROJECT}{DIR}/merge")
file="Step2_5_0_5TP_6SL_010124_010725.csv"

file_path = os.path.join(directory, file)

@jit(nopython=True)
def analyser_et_filtrer_sessions_numba(session_start_end, timestamps, duree_normale, seuil_anormal):
    sessions = []
    sessions_normales = []
    sessions_anormales = 0
    sessions_normales_count = 0
    sessions_superieures = 0
    duree_minimale = np.iinfo(np.int64).max
    duree_maximale = 0
    session_plus_courte = None
    session_plus_longue = None
    session_start = None

    for i in range(len(session_start_end)):
        if session_start_end[i] == 10:
            session_start = timestamps[i]
        elif session_start_end[i] == 20 and session_start is not None:
            session_end = timestamps[i]
            duree = (session_end - session_start) // 60  # Convertir de secondes en minutes

            # Classifier la session
            if duree < seuil_anormal * duree_normale:
                sessions_anormales += 1
            elif duree > duree_normale:
                sessions_superieures += 1
            else:
                sessions_normales_count += 1
                sessions_normales.append((session_start, session_end))

            sessions.append((session_start, session_end, duree))

            # Mettre à jour la session la plus courte et la plus longue
            if duree < duree_minimale:
                duree_minimale = duree
                session_plus_courte = (session_start, session_end, duree)
            if duree > duree_maximale:
                duree_maximale = duree
                session_plus_longue = (session_start, session_end, duree)

            session_start = None

    return sessions, sessions_normales, sessions_anormales, sessions_normales_count, sessions_superieures, session_plus_courte, session_plus_longue


def analyser_et_sauvegarder_sessions(df, duree_normale=1380, seuil_anormal=0.95, taille_lot=100000,
                                     fichier_original=None, sessions_a_sauvegarder=None):
    print_notification("Début de l'analyse et de la sauvegarde des sessions")

    if 'sc_sessionStartEnd' not in df.columns or 'timeStampOpeningConvertedtoDate' not in df.columns:
        raise ValueError(
            "Les colonnes 'sc_sessionStartEnd' et 'timeStampOpeningConvertedtoDate' doivent être présentes dans le DataFrame.")

    if not np.all(np.isin(df['sc_sessionStartEnd'], [10, 15, 20])):
        raise ValueError("La colonne 'sc_sessionStartEnd' contient des valeurs autres que 10, 15 ou 20.")

    if df['sc_sessionStartEnd'].value_counts()[10] != df['sc_sessionStartEnd'].value_counts()[20]:
        #raise ValueError("Le nombre de débuts de session (10) ne correspond pas au nombre de fins de session (20).")
        print("Le nombre de débuts de session (10) ne correspond pas au nombre de fins de session (20).")

    print_notification("Préparation des données pour l'analyse")
    timestamps = df['timeStampOpeningConvertedtoDate'].astype(np.int64).values // 10 ** 9
    session_start_end = df['sc_sessionStartEnd'].values

    all_sessions = []
    all_sessions_normales = []
    sessions_anormales = 0
    sessions_normales = 0
    sessions_superieures = 0
    session_plus_courte = None
    session_plus_longue = None

    print_notification("Début de l'analyse des sessions par lots")
    for i in range(0, len(df), taille_lot):
        lot_session_start_end = session_start_end[i:i + taille_lot]
        lot_timestamps = timestamps[i:i + taille_lot]

        results = analyser_et_filtrer_sessions_numba(lot_session_start_end, lot_timestamps, duree_normale,
                                                     seuil_anormal)

        all_sessions.extend(results[0])
        all_sessions_normales.extend(results[1])
        sessions_anormales += results[2]
        sessions_normales += results[3]
        sessions_superieures += results[4]

        if results[5] and (session_plus_courte is None or results[5][2] < session_plus_courte[2]):
            session_plus_courte = results[5]
        if results[6] and (session_plus_longue is None or results[6][2] > session_plus_longue[2]):
            session_plus_longue = results[6]

        print_notification(f"Lot traité : {i + 1} à {min(i + taille_lot, len(df))} sur {len(df)}")

    print_notification("Conversion des timestamps")
    all_sessions = [(pd.Timestamp(start, unit='s'), pd.Timestamp(end, unit='s'), pd.Timedelta(minutes=int(duree))) for
                    start, end, duree in all_sessions]
    if session_plus_courte:
        session_plus_courte = (
        pd.Timestamp(session_plus_courte[0], unit='s'), pd.Timestamp(session_plus_courte[1], unit='s'),
        pd.Timedelta(minutes=int(session_plus_courte[2])))
    if session_plus_longue:
        session_plus_longue = (
        pd.Timestamp(session_plus_longue[0], unit='s'), pd.Timestamp(session_plus_longue[1], unit='s'),
        pd.Timedelta(minutes=int(session_plus_longue[2])))

    # Sauvegarder les sessions normales si un fichier original est spécifié
    if fichier_original:
        print_notification("Début de la sauvegarde des sessions normales")
        df_sessions_normales = pd.DataFrame()

        # # Trier les sessions normales par ordre chronologique décroissant
        # all_sessions_normales.sort(key=lambda x: x[1], reverse=True)

        # Si un nombre spécifique de sessions est demandé, ne prendre que les dernières sessions
        if sessions_a_sauvegarder and sessions_a_sauvegarder < len(all_sessions_normales):
            print_notification(f"Sauvegarde des {sessions_a_sauvegarder} dernières sessions normales")
            all_sessions_normales = all_sessions_normales[:sessions_a_sauvegarder]
        else:
            print_notification("Sauvegarde de toutes les sessions normales")

        # Inverser l'ordre pour traiter du plus ancien au plus récent
        #all_sessions_normales.reverse()

        for i, (start, end) in enumerate(all_sessions_normales):
            start_ts = pd.Timestamp(start, unit='s')
            end_ts = pd.Timestamp(end, unit='s')
            session_data = df[
                (df['timeStampOpeningConvertedtoDate'] >= start_ts) & (df['timeStampOpeningConvertedtoDate'] <= end_ts)]
            df_sessions_normales = pd.concat([df_sessions_normales, session_data])
            if (i + 1) % 100 == 0:
                print_notification(f"Sessions normales traitées : {i + 1} sur {len(all_sessions_normales)}")

        df_sessions_normales = df_sessions_normales.reset_index(drop=True)

        # Modify the part where the new filename is created
        nom_fichier = os.path.basename(fichier_original)
        nom_fichier, extension = os.path.splitext(nom_fichier)
        dossier = os.path.dirname(fichier_original)

        fileStep3 = file.replace("Step2", "Step3")
        fileStep3 = os.path.splitext(fileStep3)[0]
        if sessions_a_sauvegarder:
            nouveau_fichier = f"{fileStep3}_extractOnly{sessions_a_sauvegarder}LastFullSession{extension}"
        else:
            nouveau_fichier = f"{fileStep3}_extractOnlyFullSession{extension}"

        # Combine the directory path with the new filename
        nouveau_fichier_complet = os.path.join(dossier, nouveau_fichier)

        print_notification(f"Sauvegarde du fichier : {nouveau_fichier_complet}")
        df_sessions_normales.to_csv(nouveau_fichier_complet, sep=';', index=False)

        print_notification(f"Les sessions normales ont été sauvegardées dans le fichier : {nouveau_fichier}")
        print_notification(f"Nombre de lignes dans le nouveau fichier : {len(df_sessions_normales)}")

    print_notification("Fin de l'analyse et de la sauvegarde des sessions")
    return all_sessions, sessions_anormales, sessions_normales, sessions_superieures, session_plus_courte, session_plus_longue


# Demander le nombre de sessions à sauvegarder
sessions_a_sauvegarder = input(
    "Combien de sessions voulez-vous sauvegarder ? (Appuyez sur Entrée pour toutes les sessions) : ")
sessions_a_sauvegarder = int(sessions_a_sauvegarder) if sessions_a_sauvegarder.strip() else None

# Charger les données
print_notification("Début du chargement des données")
df = pd.read_csv(file_path, sep=';')
print_notification(f"Données chargées : {len(df)} lignes")

# Convertir la colonne timeStampOpening en datetime
print_notification("Conversion de la colonne timeStampOpening en datetime")
df['timeStampOpeningConvertedtoDate'] = pd.to_datetime(df['sc_timeStampOpening'], unit='s')

# Supprimer les premières lignes avec sc_sessionStartEnd=15 en début de fichier
print_notification("Nettoyage des données initiales")
premiere_non_15 = df[df['sc_sessionStartEnd'] != 15].index[0]
df = df.iloc[premiere_non_15:].reset_index(drop=True)
print_notification(f"Nombre de lignes supprimées avec sc_sessionStartEnd=15 en début de fichier: {premiere_non_15}")

# Vérifier que le premier sc_sessionStartEnd est égal à 20
premier_non_15 = df['sc_sessionStartEnd'].iloc[0]
if premier_non_15 != 20:
    print_notification(
        f"ERREUR : Le premier sc_sessionStartEnd après suppression des 15 n'est pas 20, mais {premier_non_15}")
else:
    print_notification(
        "Le premier sc_sessionStartEnd après suppression des 15 est bien 20. Cette ligne va être supprimée.")
    df = df.iloc[1:].reset_index(drop=True)
    print_notification("La ligne avec sc_sessionStartEnd=20 a été supprimée.")

# Vérifier que la première ligne du nouveau dataframe a sc_sessionStartEnd=10
if df['sc_sessionStartEnd'].iloc[0] == 10:
    print_notification("La première ligne du nouveau dataframe a bien sc_sessionStartEnd=10")
else:
    print_notification(
        f"ERREUR : La première ligne du nouveau dataframe a sc_sessionStartEnd={df['sc_sessionStartEnd'].iloc[0]}, pas 10")

# Remplacer le sc_sessionStartEnd de la dernière ligne par 20
derniere_ligne_index = df.index[-1]
df.loc[derniere_ligne_index, 'sc_sessionStartEnd'] = 20
print_notification(f"Le sc_sessionStartEnd de la dernière ligne (index {derniere_ligne_index}) a été remplacé par 20")

# Analyser les sessions et sauvegarder les sessions normales
try:
    print_notification("Début de l'analyse des sessions")
    sessions, sessions_anormales, sessions_normales, sessions_superieures, session_plus_courte, session_plus_longue = analyser_et_sauvegarder_sessions(
        df, fichier_original=file_path, sessions_a_sauvegarder=sessions_a_sauvegarder)

    print_notification("Résultats de l'analyse :")
    print(f"\nNombre total de sessions : {len(sessions)}")
    print(f"Nombre de sessions normales : {sessions_normales}")
    print(f"Nombre de sessions anormales : {sessions_anormales}")
    print(f"Nombre de sessions supérieures à la durée normale : {sessions_superieures}")

    if session_plus_courte:
        start, end, duree = session_plus_courte
        print(f"\nSession la plus courte :")
        print(f"  Début : {start}")
        print(f"  Fin : {end}")
        print(f"  Durée : {duree}")

    if session_plus_longue:
        start, end, duree = session_plus_longue
        print(f"\nSession la plus longue :")
        print(f"  Début : {start}")
        print(f"  Fin : {end}")
        print(f"  Durée : {duree}")

except ValueError as e:
    print_notification(f"Erreur lors de l'analyse des sessions : {e}")

print_notification("Fin du script")