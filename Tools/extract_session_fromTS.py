import pandas as pd
import os
from standardFunc_sauv import print_notification, load_data


def extract_data_by_timestamp(df, start_ts, end_ts, output_directory, input_filename):
    # Filtrer les données entre les timestamps
    mask = (df['timeStampOpening'] >= start_ts) & (df['timeStampOpening'] <= end_ts)
    filtered_df = df[mask]

    # Créer le nom du fichier de sortie
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_filtered_{start_ts}_{end_ts}.csv"
    output_path = os.path.join(output_directory, output_filename)

    # Sauvegarder dans un nouveau fichier CSV avec délimiteur ";"
    filtered_df.to_csv(output_path, index=False, sep=';')
    print(f"Données extraites sauvegardées dans : {output_path}")
    print(f"Nombre de lignes extraites : {len(filtered_df)}")

    return filtered_df


# Définition des chemins et fichiers
file_name = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat.csv"
# file_name = "Step3_4_0_4TP_1SL_080919_091024_extractOnly220LastFullSession.csv"

# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_6TP_1SL\\merge"

# Combiner le chemin du répertoire avec le nom du fichier
file_path = os.path.join(directory_path, file_name)

# Charger les données
df = load_data(file_path)

# Paramètres pour l'extraction
start_timestamp = 1652196157
end_timestamp = 1652294677

# Extraire les données et sauvegarder
extracted_data = extract_data_by_timestamp(
    df=df,
    start_ts=start_timestamp,
    end_ts=end_timestamp,
    output_directory=directory_path,
    input_filename=file_name
)