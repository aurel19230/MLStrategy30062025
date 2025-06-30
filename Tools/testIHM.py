import pandas as pd
import numpy as np
import numba as nb
import time

@nb.njit
def calculate_session_duration(session_start_end, delta_timestamp):
    total_minutes = 0
    residual_minutes = 0
    session_start = -1
    complete_sessions = 0
    in_session = False

    # Gérer le cas où le fichier ne commence pas par un 10
    if session_start_end[0] != 10:
        first_20_index = np.where(session_start_end == 20)[0][0]
        residual_minutes += delta_timestamp[first_20_index] - delta_timestamp[0]

    for i in range(len(session_start_end)):
        if session_start_end[i] == 10:
            session_start = i
            in_session = True
        elif session_start_end[i] == 20:
            if in_session:
                # Session complète
                total_minutes += delta_timestamp[i] - delta_timestamp[session_start]
                complete_sessions += 1
                in_session = False
            else:
                # 20 sans 10 précédent, ne devrait pas arriver mais gérons le cas
                residual_minutes += delta_timestamp[i] - delta_timestamp[session_start if session_start != -1 else 0]

    # Gérer le cas où le fichier se termine au milieu d'une session
    if in_session:
        residual_minutes += delta_timestamp[-1] - delta_timestamp[session_start]
    elif session_start_end[-1] != 20:
        # Si la dernière valeur n'est pas 20 et qu'on n'est pas dans une session,
        # ajoutons le temps depuis le dernier 20 jusqu'à la fin
        last_20_index = np.where(session_start_end == 20)[0][-1]
        residual_minutes += delta_timestamp[-1] - delta_timestamp[last_20_index]

    return complete_sessions, total_minutes, residual_minutes

def calculate_and_display_sessions(df):
    session_start_end = df['SessionStartEnd'].astype(np.int32).values
    delta_timestamp = df['deltaTimestampOpening'].astype(np.float64).values

    complete_sessions, total_minutes, residual_minutes = calculate_session_duration(session_start_end, delta_timestamp)

    session_duration_hours = 23
    session_duration_minutes = session_duration_hours * 60

    residual_sessions = residual_minutes / session_duration_minutes
    total_sessions = complete_sessions + residual_sessions

    #print(f"Nombre de sessions complètes : {complete_sessions}")
    #print(f"Minutes résiduelles : {residual_minutes:.2f}")
    #print(f"Équivalent en sessions des minutes résiduelles : {residual_sessions:.2f}")
    #print(f"Nombre total de sessions (complètes + résiduelles) : {total_sessions:.2f}")

    return total_sessions

# Chemin vers votre fichier CSV
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_04102024\merge\Step2_MergedAllFile_Step1_4_merged.csv"

try:
    # Lecture du fichier CSV
    print("Lecture du fichier CSV...")
    df = pd.read_csv(file_path, delimiter=';')
    print("Fichier CSV lu avec succès.")

    # Calcul et affichage des informations sur les sessions
    print("\nAnalyse des sessions de trading :")
    start_time = time.time()
    total_sessions = calculate_and_display_sessions(df)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\nTemps d'exécution de calculate_and_display_sessions : {execution_time:.4f} secondes")
    # Affichage de quelques statistiques supplémentaires
    print("\nStatistiques supplémentaires :")
    print(f"Nombre total de lignes dans le fichier : {len(df)}")
    print(f"Nombre de valeurs uniques dans SessionStartEnd : {df['SessionStartEnd'].nunique()}")
    print("Valeurs uniques dans SessionStartEnd et leur fréquence :")
    print(df['SessionStartEnd'].value_counts().sort_index())

except Exception as e:
    print(f"Une erreur s'est produite : {e}")