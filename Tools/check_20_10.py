import pandas as pd

def validate_session_start_end(df):
    """
    Vérifie que chaque occurrence de 20 dans SessionStartEnd
    est suivie par un 10. Lève une erreur sinon.
    """
    session_values = df['SessionStartEnd'].values
    for i in range(len(session_values) - 1):
        if session_values[i] == 20:
            if session_values[i + 1] != 10:
                raise ValueError(
                    f"Erreur : SessionStartEnd=20 à l'index {i} n'est pas suivi par un 10 à l'index {i + 1}."
                )
    print("Validation réussie : chaque 20 est bien suivi d'un 10.")
import os
# Exemple d'utilisation
if __name__ == "__main__":
    FILE_NAME = "Step2_4_0_5TP_1SL_newBB_080919_281124.csv"
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_5TP_1SL_newBB\merge"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME)

    df = pd.read_csv(FILE_PATH, sep=";",  encoding="ISO-8859-1")

    try:
        validate_session_start_end(df)
    except ValueError as e:
        print(e)
