import pandas as pd
from standardFunc_sauv import load_data, split_sessions, print_notification
from decimal import Decimal

def identify_sessions(df):
    session_col = "SessionStartEnd"
    if session_col not in df.columns:
        raise ValueError(f"La colonne {session_col} n'existe pas dans le DataFrame.")

    session_starts = df[df[session_col] == 10].index
    session_ends = df[df[session_col] == 20].index

    print(f"Nombre de débuts de session détectés : {len(session_starts)}")
    print(f"Nombre de fins de session détectées : {len(session_ends)}")

    sessions = []
    for start, end in zip(session_starts, session_ends):
        if start < end:
            session = df.loc[start:end]
            if session.iloc[0][session_col] == 10 and session.iloc[-1][session_col] == 20:
                sessions.append(session)

    total_sessions = len(sessions)
    print(f"Nombre de sessions complètes et valides extraites : {total_sessions}")

    return total_sessions, sessions

def calculate_compatible_n_splits(total_sessions, test_size, min_train_sessions=2, min_test_sessions=2):
    test_size_decimal = Decimal(str(test_size))
    test_sessions = int(total_sessions * test_size_decimal)
    train_sessions = total_sessions - test_sessions

    if test_sessions < min_test_sessions or train_sessions < min_train_sessions:
        return []

    compatible_splits = []
    for n_splits in range(2, train_sessions):
        if train_sessions % (n_splits + 1) == 0:  # Vérifier la divisibilité par (n_splits + 1)
            calculated_test_size = Decimal(test_sessions) / Decimal(total_sessions)
            if len(str(calculated_test_size).split('.')[-1]) <= 2:
                if calculated_test_size == test_size_decimal:
                    compatible_splits.append(n_splits)

    return compatible_splits

# Charger les données
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge13092024\Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnly100LastFullSession_feat_winsorizedScaledWithNanVal.csv"
try:
    df = load_data(file_path)
    print("Fichier chargé avec succès.")
except ValueError as e:
    print(f"Erreur lors du chargement du fichier : {e}")
    exit(1)

# Identifier et compter les sessions
try:
    total_sessions, sessions = identify_sessions(df)
except ValueError as e:
    print(f"Erreur : {e}")
    exit(1)

if total_sessions == 0:
    print("Aucune session complète et valide détectée dans les données.")
    exit(1)

# Demander le test_size souhaité
test_size = float(input("Entrez le test_size souhaité (par exemple, 0.25 pour 25%) : "))

# Calculer les n_splits compatibles
compatible_splits = calculate_compatible_n_splits(total_sessions, test_size)

if not compatible_splits:
    print(f"Aucune configuration de n_splits compatible trouvée pour test_size = {test_size}.")
    print("Assurez-vous que le test_size a au maximum 2 chiffres après la virgule.")
    exit(1)

# Vérifier que les splits donnent des sessions complètes
train_df, test_df = split_sessions(df, test_size=test_size, min_train_sessions=2, min_test_sessions=2)
unique_sessions = train_df[train_df['SessionStartEnd'] == 10].index
train_total_sessions = len(unique_sessions)

print(f"\nNombre total de sessions dans train_df: {train_total_sessions}")

valid_splits = []
for n_splits in compatible_splits:
    block_size = train_total_sessions // (n_splits + 1)
    if train_total_sessions % (n_splits + 1) == 0:
        valid_splits.append(n_splits)
        print(f"n_splits = {n_splits} (Train: {train_total_sessions} sessions, {block_size} sessions par bloc)")
    else:
        print(f"n_splits = {n_splits} (incompatible: ne donne pas des sessions complètes)")

if not valid_splits:
    print("Aucun n_splits ne donne des sessions complètes. Veuillez essayer un autre test_size.")
    exit(1)

print("\nChoisissez un n_splits dans cette liste pour que la validation croisée fonctionne correctement sans sessions partielles.")