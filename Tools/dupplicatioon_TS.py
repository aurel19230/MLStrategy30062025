import pandas as pd
import os


def charger_donnees(chemin_fichier):
    """Charge les données depuis un fichier CSV et formate le timeStampOpening."""
    print(f"Chargement des données depuis {os.path.basename(chemin_fichier)}...")
    df = pd.read_csv(chemin_fichier, sep=";", encoding="ISO-8859-1")
    # Convertir timeStampOpening en entier
    df['timeStampOpening'] = df['timeStampOpening'].astype('Int64')
    return df


def configurer_pandas():
    """Configure les options d'affichage de pandas sans aucune limitation."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 10)
    pd.set_option('display.float_format', lambda x: '%.10f' % x if abs(x) < 1e10 else '%.0f' % x)


def nettoyer_dernieres_lignes(df):
    """
    Supprime les dernières lignes du DataFrame à partir du dernier SessionStartEnd=10.

    Args:
        df: DataFrame à nettoyer
    Returns:
        DataFrame nettoyé
    """
    # Trouver l'index de la dernière occurrence de SessionStartEnd=10
    dernier_index = df.loc[df['SessionStartEnd'] == 10].index[-1]

    # Garder seulement les lignes jusqu'à cet index (non inclus)
    df_nettoye = df.loc[:dernier_index - 1].copy()

    print(f"Nombre de lignes supprimées : {len(df) - len(df_nettoye)}")
    return df_nettoye


def verifier_doublons(df1, df2=None, colonnes_a_verifier=None):
    """
    Vérifie les doublons dans un ou deux DataFrames.

    Args:
        df1: Premier DataFrame
        df2: Deuxième DataFrame (optionnel)
        colonnes_a_verifier: Liste des colonnes à vérifier pour les doublons
    """
    if colonnes_a_verifier is None:
        colonnes_a_verifier = ['timeStampOpening', 'close', 'open', 'high', 'low',
                               'volume', 'atr', 'vaDelta_6periods', 'vaVol_16periods', "perctBB", "SessionStartEnd"]

    if df2 is None:
        # Vérification des doublons dans un seul fichier
        doublons = df1[df1.duplicated(subset=colonnes_a_verifier, keep=False)]
        print("\nDoublons trouvés dans le fichier :")
        print(f"Nombre de lignes en double : {len(doublons)}")
        if len(doublons) > 0:
            print("\nDétail des doublons :")
            with pd.option_context('display.float_format', lambda x: '%.0f' % x if x >= 1e10 else '%.10f' % x):
                # Sélectionner uniquement les colonnes spécifiées
                print(doublons[colonnes_a_verifier].sort_values(by=colonnes_a_verifier))
    else:
        # Vérification des doublons entre deux fichiers
        df_combine = pd.concat([df1, df2])
        doublons = df_combine[df_combine.duplicated(subset=colonnes_a_verifier, keep=False)]
        print("\nDoublons trouvés entre les deux fichiers :")
        print(f"Nombre de lignes en double : {len(doublons)}")
        if len(doublons) > 0:
            print("\nDétail des doublons :")
            with pd.option_context('display.float_format', lambda x: '%.0f' % x if x >= 1e10 else '%.10f' % x):
                # Sélectionner uniquement les colonnes spécifiées
                print(doublons[colonnes_a_verifier].sort_values(by=colonnes_a_verifier))


def main():
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge\splitandchek"
    FILE_NAME = "Step1_050522_111223_4TicksRev_Cleaned_2.csv"
    FILE_NAME1 = "Step1_101223_141024_4TicksRev_Cleaned_3.csv"

    # Configuration de pandas
    configurer_pandas()

    # Demander à l'utilisateur le mode de vérification
    choix = input(
        "Appuyez sur 'd' pour vérifier les doublons entre les deux fichiers, ou une autre touche pour vérifier uniquement le premier fichier : ").lower()

    try:
        # Charger le premier fichier
        df1 = charger_donnees(os.path.join(DIRECTORY_PATH, FILE_NAME))
        print(f"Taille initiale du premier fichier : {df1.shape}")

        # Nettoyer les dernières lignes du premier fichier
        df1 = nettoyer_dernieres_lignes(df1)
        print(f"Taille du premier fichier après nettoyage : {df1.shape}")

        if choix == 'd':
            # Charger le deuxième fichier et vérifier les doublons entre les deux
            df2 = charger_donnees(os.path.join(DIRECTORY_PATH, FILE_NAME1))
            print(f"Taille du deuxième fichier : {df2.shape}")
            verifier_doublons(df1, df2)
        else:
            # Vérifier les doublons dans le premier fichier uniquement
            verifier_doublons(df1)

    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")


if __name__ == "__main__":
    main()