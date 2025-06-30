import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime, timedelta

# Configuration
CONFIG = {
    'NUM_GROUPS': 9,
    'MIN_RANGE': 30,  # en minutes
    'FILE_PATH': r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\\4_0_6TP_1SL\merge\Step3_4_0_6TP_1SL_080919_141024_extractOnlyFullSession.csv",
    'TRADING_START_TIME': "22:00",
    'FIGURE_SIZE': (20, 10),
    'GRID_ALPHA': 0.7,
}

# Définition des sections personnalisées
CUSTOM_SECTIONS = [
    {"name": "preAsian", "start": 0, "end": 240, "index": 0},
    {"name": "asianAndPreEurop", "start": 240, "end": 540, "index": 1},
    {"name": "europMorning", "start": 540, "end": 810, "index": 2},
    {"name": "europLunch", "start": 810, "end": 870, "index": 3},
    {"name": "preUS", "start": 870, "end": 930, "index": 4},
    {"name": "usMoning", "start": 930, "end": 1065, "index": 5},
    {"name": "usAfternoon", "start": 1065, "end": 1200, "index": 6},
    {"name": "usEvening", "start": 1200, "end": 1290, "index": 7},
    {"name": "usEnd", "start": 1290, "end": 1335, "index": 8},
    {"name": "closing", "start": 1335, "end": 1380, "index": 9},
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        file_path (str): Chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    return pd.read_csv(file_path, sep=';')


def get_custom_section(minutes: int) -> dict:
    """
    Détermine la section personnalisée correspondant à un nombre de minutes donné.

    Args:
        minutes (int): Nombre de minutes depuis le début de la session.

    Returns:
        dict: Informations sur la section personnalisée.
    """
    for section in CUSTOM_SECTIONS:
        if section['start'] <= minutes < section['end']:
            return section
    return CUSTOM_SECTIONS[-1]  # Retourne la dernière section si aucune correspondance n'est trouvée


def preprocess_data(df: pd.DataFrame, min_range: int) -> pd.DataFrame:
    """
    Prétraite les données en calculant les sections de temps et les sections personnalisées.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        min_range (int): Intervalle de temps en minutes pour chaque section.

    Returns:
        pd.DataFrame: DataFrame prétraité avec les sections de temps calculées.
    """

    df['deltaTimestampOpeningSection5min'] = df['deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 5)) * 5, 1350))

    # Créer un index consécutif pour les sections de 5 minutes
    unique_sections = sorted(df['deltaTimestampOpeningSection5min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    df['deltaTimestampOpeningSection5index'] = df['deltaTimestampOpeningSection5min'].map(section_to_index)


    df['deltaTimestampOpeningSection30min'] = df['deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / min_range)) * min_range, 1350))

    # Créer un index consécutif pour les sections de 30 minutes
    unique_sections = sorted(df['deltaTimestampOpeningSection30min'].unique())
    section_to_index = {section: index for index, section in enumerate(unique_sections)}
    df['deltaTimestampOpeningSection30index'] = df['deltaTimestampOpeningSection30min'].map(section_to_index)

    # Calculer deltaCustomSectionMin
    df['deltaCustomSectionMin'] = df['deltaTimestampOpening'].apply(
        lambda x: get_custom_section(x)['start'])

    # Créer un index consécutif pour les sections personnalisées
    unique_custom_sections = sorted(df['deltaCustomSectionMin'].unique())
    custom_section_to_index = {section: index for index, section in enumerate(unique_custom_sections)}
    df['deltaCustomSectionIndex'] = df['deltaCustomSectionMin'].map(custom_section_to_index)

    return df


@jit(nopython=True)
def calculate_volume_change(volumes: np.ndarray) -> np.ndarray:
    """
    Calcule la variation de volume entre les sections consécutives.

    Args:
        volumes (np.ndarray): Array des volumes.

    Returns:
        np.ndarray: Array des variations de volume.
    """
    volume_change = np.zeros(len(volumes))
    for i in range(1, len(volumes)):
        if volumes[i - 1] != 0:
            volume_change[i] = (volumes[i] - volumes[i - 1]) / volumes[i - 1]
        else:
            volume_change[i] = 0
    return volume_change


@jit(nopython=True)
def create_groups(volumes: np.ndarray, volume_change: np.ndarray, num_groups: int) -> list:
    """
    Crée des groupes basés sur les variations de volume.

    Args:
        volumes (np.ndarray): Array des volumes.
        volume_change (np.ndarray): Array des variations de volume.
        num_groups (int): Nombre de groupes à créer.

    Returns:
        list: Liste des groupes créés.
    """
    n = len(volumes)
    if n < num_groups:
        return [list(range(n))]

    top_variations = np.argsort(np.abs(volume_change))[-(num_groups - 1):]
    top_variations = np.sort(top_variations)

    groups = []
    start = 0
    for end in top_variations:
        if end > start:
            groups.append(list(range(start, end)))
            start = end

    if start < n:
        groups.append(list(range(start, n)))

    while len(groups) < num_groups:
        if len(groups) <= 1:
            break
        min_size = n + 1
        min_index = 0
        for i in range(len(groups) - 1):
            size = len(groups[i]) + len(groups[i + 1])
            if size < min_size:
                min_size = size
                min_index = i
        groups[min_index] = groups[min_index] + groups[min_index + 1]
        groups.pop(min_index + 1)

    return groups


def minutes_to_time(minutes: int) -> str:
    """
    Convertit les minutes en format d'heure lisible.

    Args:
        minutes (int): Nombre de minutes depuis le début de la journée de trading.

    Returns:
        str: Heure au format HH:MM.
    """
    start_time = datetime.strptime(CONFIG['TRADING_START_TIME'], "%H:%M")
    new_time = start_time + timedelta(minutes=int(minutes))
    return new_time.strftime("%H:%M")


def print_group_info(groups: list, times: np.ndarray, volumes: np.ndarray):
    """
    Affiche les informations pour chaque groupe.

    Args:
        groups (list): Liste des groupes.
        times (np.ndarray): Array des temps.
        volumes (np.ndarray): Array des volumes.
    """
    print(f"Nombre de groupes créés : {len(groups)}")
    total_volume = np.sum(volumes)

    for i, group in enumerate(groups, 1):
        if group:
            start_time = times[group[0]]
            end_time = times[group[-1]] + CONFIG['MIN_RANGE']
            start_time_str = minutes_to_time(start_time)
            end_time_str = minutes_to_time(end_time)
            group_volume = np.sum(volumes[group])
            percentage = (group_volume / total_volume) * 100

            print(f"\nGroupe {i}:")
            print(f"  De {start_time} minutes ({start_time_str}) à {end_time} minutes ({end_time_str})")
            print(f"  Volume total = {group_volume}")
            print(f"  Pourcentage du volume total = {percentage:.2f}%")
        else:
            print(f"\nGroupe {i}: Vide")


def plot_volume_distribution(times: np.ndarray, volumes: np.ndarray, groups: list):
    """
    Trace la distribution des volumes avec les groupes.

    Args:
        times (np.ndarray): Array des temps.
        volumes (np.ndarray): Array des volumes.
        groups (list): Liste des groupes.
    """
    plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    plt.bar(times, volumes, width=CONFIG['MIN_RANGE'], align='edge')
    plt.title(f'Volume par section de {CONFIG["MIN_RANGE"]} minutes avec {CONFIG["NUM_GROUPS"]} groupes')
    plt.xlabel('Temps (minutes depuis ' + CONFIG['TRADING_START_TIME'] + ')')
    plt.ylabel('Volume')

    for group in groups[1:]:
        if group:
            plt.axvline(x=times[group[0]], color='r', linestyle='--', linewidth=2)

    xticks = np.arange(0, 1440, CONFIG['MIN_RANGE'])
    xlabels = [minutes_to_time(x) for x in xticks]
    plt.xticks(xticks, xlabels, rotation=90, ha='center', fontsize=8)

    plt.xlim(0, 1440)
    plt.grid(axis='y', linestyle='--', alpha=CONFIG['GRID_ALPHA'])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def plot_custom_sections_volume(df: pd.DataFrame):
    """
    Trace la distribution des volumes par sections personnalisées.

    Args:
        df (pd.DataFrame): DataFrame contenant les données de volume et les sections personnalisées.
    """
    volume_by_custom_section = df.groupby('deltaCustomSectionIndex')['volume'].sum().reset_index()

    plt.figure(figsize=CONFIG['FIGURE_SIZE'])
    bars = plt.bar(volume_by_custom_section['deltaCustomSectionIndex'],
                   volume_by_custom_section['volume'],
                   align='center')

    plt.title('Volume par section personnalisée')
    plt.xlabel('Section')
    plt.ylabel('Volume')

    # Ajouter les noms des sections sur l'axe x
    plt.xticks(range(len(CUSTOM_SECTIONS)),
               [section['name'] for section in CUSTOM_SECTIONS],
               rotation=45, ha='right')

    # Ajouter les valeurs de volume au-dessus de chaque barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,.0f}',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=CONFIG['GRID_ALPHA'])
    plt.show()


def plot_custom_sections_with_15min_volume(df: pd.DataFrame):
    """
    Trace la distribution des volumes par tranches de 15 minutes avec les sections personnalisées en arrière-plan.

    Args:
        df (pd.DataFrame): DataFrame contenant les données de volume et les sections personnalisées.
    """
    # Calculer les tranches de 15 minutes
    df['deltaTimestampOpeningSection15min'] = df['deltaTimestampOpening'].apply(
        lambda x: min(int(np.floor(x / 15)) * 15, 1380))

    volume_by_15min = df.groupby('deltaTimestampOpeningSection15min')['volume'].sum().reset_index()

    plt.figure(figsize=(24, 12))  # Augmenter légèrement la taille pour plus de clarté
    bars = plt.bar(volume_by_15min['deltaTimestampOpeningSection15min'],
                   volume_by_15min['volume'],
                   width=15, align='edge')

    plt.title('Volume par tranche de 15 minutes avec sections personnalisées')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Volume')

    # Ajouter des lignes verticales et des annotations pour chaque section personnalisée
    for section in CUSTOM_SECTIONS:
        plt.axvline(x=section['start'], color='r', linestyle='--', linewidth=1)
        plt.text(section['start'], plt.ylim()[1], section['name'],
                 rotation=90, va='top', ha='right', fontsize=8)

    # Personnaliser l'axe x
    xticks = np.arange(0, 1440, 60)  # Toutes les heures
    xlabels = [f"{(22 + i) % 24:02d}:00" for i in range(24)]
    plt.xticks(xticks, xlabels, rotation=45, ha='right')

    # Ajouter des ticks mineurs pour les tranches de 15 minutes
    minor_xticks = np.arange(0, 1440, 15)
    plt.gca().set_xticks(minor_xticks, minor=True)

    plt.xlim(0, 1380)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', which='minor', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


def display_column_info(df: pd.DataFrame):
    """
    Affiche les 5 premières et 5 dernières lignes des colonnes spécifiées.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    columns_to_display = [
        'deltaTimestampOpening',
        'deltaTimestampOpeningSection5min',
        'deltaTimestampOpeningSection5index',
        'deltaTimestampOpeningSection30min',
        'deltaTimestampOpeningSection30index',
        'deltaCustomSectionMin',
        'deltaCustomSectionIndex'
    ]

    print("\nAffichage des 5 premières lignes:")
    print(df[columns_to_display].head().to_string(index=False))

    print("\nAffichage des 5 dernières lignes:")
    print(df[columns_to_display].tail().to_string(index=False))

def main():
    """
    Fonction principale qui orchestre l'analyse des volumes.
    """
    df = load_data(CONFIG['FILE_PATH'])
    df = preprocess_data(df, CONFIG['MIN_RANGE'])

    # Afficher les informations sur les colonnes spécifiées
    display_column_info(df)

    # Utiliser deltaTimestampOpeningSection30index pour grouper les données
    volume_by_section = df.groupby('deltaTimestampOpeningSection30index')['volume'].sum().reset_index()
    volume_by_section = volume_by_section.sort_values('deltaTimestampOpeningSection30index')

    # Convertir les index en minutes pour l'affichage
    times = volume_by_section['deltaTimestampOpeningSection30index'].values * CONFIG['MIN_RANGE']
    volumes = volume_by_section['volume'].values

    volume_change = calculate_volume_change(volumes)
    groups = create_groups(volumes, volume_change, CONFIG['NUM_GROUPS'])

    print_group_info(groups, times, volumes)
    plot_volume_distribution(times, volumes, groups)
    plot_custom_sections_volume(df)
    plot_custom_sections_with_15min_volume(df)

if __name__ == "__main__":
    main()