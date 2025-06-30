import re
import os
import csv
from datetime import datetime


def parse_raw_file(file_path, expected_date=None):
    """
    Parse le fichier brouillon et extrait les informations pour chaque ticker.

    Args:
        file_path: Chemin vers le fichier à analyser
        expected_date: Date attendue dans le format JJ/MM/AAAA, pour vérification
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Afficher les 200 premiers caractères pour diagnostic
    print("\nDébut du contenu du fichier:")
    print(content[:min(200, len(content))])
    print("...")

    # Identifier les tickers dans le fichier
    # Recherche les patterns étendus pour supporter tous les types de tickers
    ticker_patterns = [
        r'([A-Z]{2,3})\s*m25',  # Pour les contrats en "m25" (ESm25, NQm25, etc.)
        r'([A-Z]{2,3})\s*k25',  # Pour les contrats en "k25" (CLk25, etc.)
        r'^(6[a-zA-Z])',  # Pour les forex "6e", "6j", "6N", "6S", "6a", etc.
        r'^(GC)\s*m25',  # Spécifiquement pour GCm25
        r'^(NG)\s*k25',  # Spécifiquement pour NG k25
        r'^(RTY)\s*m25',  # Spécifiquement pour RTY m25
        r'^(YM)\s*m25',  # Spécifiquement pour YMm25
        r'^(VIX)',  # Pour VIX
        r'^(VXN)',  # Pour VXN
        r'^(RVX)',  # Pour RVX
        r'^(CL)\s*m25'  # Pour CLm25
    ]

    tickers_with_position = []

    # Identification des positions de départ de chaque ticker
    lines = content.split('\n')
    current_position = 0

    for line in lines:
        line_stripped = line.strip()
        # Si la ligne est vide, continuer
        if not line_stripped:
            current_position += len(line) + 1  # +1 pour le \n
            continue

        # Vérifier si la ligne commence par un ticker connu
        for pattern in ticker_patterns:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                ticker_name = match.group(0)
                tickers_with_position.append((ticker_name, current_position))
                break

        current_position += len(line) + 1  # +1 pour le \n

    # Si l'approche par ligne ne donne pas de bons résultats, utiliser la recherche dans tout le texte
    if not tickers_with_position:
        for pattern in ticker_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                ticker = match.group(0)
                # Vérifier si ce ticker est au début d'une ligne
                line_start = content.rfind('\n', 0, match.start()) + 1
                if line_start == match.start() or line_start == 0:
                    tickers_with_position.append((ticker, match.start()))

    # Trier les tickers par leur position dans le texte
    tickers_with_position.sort(key=lambda x: x[1])

    print(f"Tickers trouvés: {[t[0] for t in tickers_with_position]}")

    all_data = []
    errors = []

    # Date du jour (à partir du nom de fichier ou actuelle si non disponible)
    if expected_date:
        current_date = expected_date
    else:
        current_date = datetime.now().strftime('%d/%m/%Y')

    # Pour chaque ticker trouvé, extraire son bloc de données
    for i, (ticker, ticker_start_pos) in enumerate(tickers_with_position):
        # Déterminer le type de ticker
        if "m25" in ticker.lower():
            ticker_type = "m25"
            base_ticker = re.match(r'([A-Z]+)', ticker).group(1) if re.match(r'([A-Z]+)', ticker) else ticker.replace(
                "m25", "").strip()
        elif "k25" in ticker.lower():
            ticker_type = "k25"
            base_ticker = re.match(r'([A-Z]+)', ticker).group(1) if re.match(r'([A-Z]+)', ticker) else ticker.replace(
                "k25", "").strip()
        elif ticker.startswith("6"):
            ticker_type = "forex"
            base_ticker = ticker
        elif ticker in ["VIX", "VXN", "RVX"]:
            ticker_type = "volatility"
            base_ticker = ticker
        else:
            # Cas par défaut
            ticker_type = "other"
            base_ticker = ticker

        # Déterminer la fin du bloc (soit le début du prochain ticker, soit la fin du contenu)
        next_ticker_pos = len(content)
        if i < len(tickers_with_position) - 1:
            next_ticker_pos = tickers_with_position[i + 1][1]

        # Extraire le bloc de texte pour ce ticker
        block = content[ticker_start_pos:next_ticker_pos].strip()
        print(f"\nBloc pour {ticker} (100 premiers caractères): {block[:min(100, len(block))]}")

        # Liste pour enregistrer les éléments manquants
        missing_elements = []

        # Extraire le gamma (Français et Anglais) - PATTERNS AMÉLIORÉS
        gamma_patterns = [
            # Patterns en français avec plus de flexibilité
            r'gamma\s+(positif|n[ée]gatif)',
            r'gamma\s*(positif|n[ée]gatif)',
            r'\sgamma\s+(positif|n[ée]gatif)',  # Avec espace avant
            r'\sgamma\s*(positif|n[ée]gatif)',  # Avec espace avant
            r'gamma.*?(positif|n[ée]gatif)',  # Tout ce qui se trouve entre
            # Patterns en anglais
            r'gamma\s+(positive|negative)',
            r'gamma\s*(positive|negative)',
            r'\sgamma\s+(positive|negative)',
            r'\sgamma\s*(positive|negative)',
            r'gamma.*?(positive|negative)',
        ]

        gamma = ""
        for pattern in gamma_patterns:
            gamma_match = re.search(pattern, block, re.IGNORECASE)
            if gamma_match:
                gamma_val = gamma_match.group(1).lower()
                # Standardisation entre français et anglais
                if gamma_val in ["positive", "positif"]:
                    gamma = "positif"
                elif gamma_val in ["negative", "negatif", "négatif"]:
                    gamma = "negatif"
                break

        # Si toujours pas trouvé, essayer des recherches simples de mots-clés
        if not gamma:
            lower_block = block.lower()
            if "gamma positif" in lower_block or "gamma positive" in lower_block:
                gamma = "positif"
            elif "gamma negatif" in lower_block or "gamma négatif" in lower_block or "gamma negative" in lower_block:
                gamma = "negatif"
            # Détecter même sans le mot 'gamma' si c'est la seule info
            elif "positif" in lower_block and "negatif" not in lower_block and "négatif" not in lower_block and "negative" not in lower_block:
                gamma = "positif"
            elif "negatif" in lower_block or "négatif" in lower_block or "negative" in lower_block:
                gamma = "negatif"

        # NOUVELLES VÉRIFICATIONS pour format "Positive Gamma" (au lieu de "gamma positif")
        if not gamma:
            gamma_patterns_extended = [
                r'[Pp]ositive\s+[Gg]amma',  # Format "Positive Gamma"
                r'[Nn]egative\s+[Gg]amma',  # Format "Negative Gamma"
                r'[Pp]ositif\s+[Gg]amma',  # Format "Positif Gamma"
                r'[Nn][ée]gatif\s+[Gg]amma'  # Format "Negatif Gamma"
            ]

            # Rechercher ces patterns spécifiques
            for pattern in gamma_patterns_extended:
                gamma_match = re.search(pattern, block, re.IGNORECASE)
                if gamma_match:
                    if "positive" in gamma_match.group(0).lower() or "positif" in gamma_match.group(0).lower():
                        gamma = "positif"
                    else:
                        gamma = "negatif"
                    break

        # Méthode de dernier recours - vérifier chaque ligne individuellement
        if not gamma:
            for line in block.split('\n'):
                line_lower = line.lower()
                if "gamma" in line_lower:
                    if "positive" in line_lower or "positif" in line_lower:
                        gamma = "positif"
                        break
                    elif "negative" in line_lower or "negatif" in line_lower or "négatif" in line_lower:
                        gamma = "negatif"
                        break

        # Gestion spéciale pour les forex (6e, 6j) et indexes de volatilité
        # Pour ces types, le gamma est souvent omis, on peut l'initialiser à une valeur par défaut
        if (ticker.startswith("6") or ticker in ["VIX", "VXN", "RVX"]) and not gamma:
            gamma = "non_specifie"  # Valeur par défaut pour les tickers sans gamma explicite

        # Vérifier si le gamma est une valeur valide
        if gamma == "négatif":
            gamma = "negatif"
        if not gamma:
            missing_elements.append("Gamma")

        # Extraire Max 1D et ses valeurs additionnelles (event et extreme)
        max_1d_patterns = [
            r'Max\s+1D\s+(\d+\.?\d*)',
            r'Max\s+1D\s*(\d+\.?\d*)',  # Sans espace après 1D
            # Patterns en anglais
            r'Max\s+1D\s+(\d+\.?\d*)',
            r'Maximum\s+1D\s+(\d+\.?\d*)'
        ]
        max_1d = ""
        for pattern in max_1d_patterns:
            max_1d_match = re.search(pattern, block)
            if max_1d_match:
                max_1d = max_1d_match.group(1)
                break
        if not max_1d:
            missing_elements.append("Max_1D")

        # Extraire Max 1D event - avec patterns améliorés (français et anglais)
        max_event_patterns = [
            # Patterns en français
            r'en cas d[\'e]?event\s+(\d+\.?\d*)',
            r'en cas d\s*event\s+(\d+\.?\d*)',  # Avec espace entre 'd' et 'event'
            r'\(en cas d[\'e]?event\s+(\d+\.?\d*)',  # Format avec parenthèses
            r'\(event\s+(\d+\.?\d*)\)',  # Format "(event X.XXX)"
            r'\(?event\s+(\d+\.?\d*)',  # Format "event X.XXX" ou "(event X.XXX"
            # Patterns en anglais
            r'in case of event\s+(\d+\.?\d*)',
            r'\(in case of event\s+(\d+\.?\d*)',
            r'\(event\s+(\d+\.?\d*)\)',
            r'event\s+(\d+\.?\d*)'
        ]
        max_1d_event = ""
        for pattern in max_event_patterns:
            # Rechercher dans la section après "Max 1D" et avant "Min 1D" si possible
            max_section = block
            if "Max 1D" in block and "Min 1D" in block:
                max_start = block.find("Max 1D")
                min_start = block.find("Min 1D")
                if max_start < min_start:
                    max_section = block[max_start:min_start]

            max_1d_event_match = re.search(pattern, max_section)
            if max_1d_event_match:
                max_1d_event = max_1d_event_match.group(1)
                break

        # Extraire Max 1D extreme
        max_extreme_patterns = [
            r'extreme\s+(\d+\.?\d*)',
            r'\(?extreme\s+(\d+\.?\d*)'  # Format avec parenthèse optionnelle
        ]
        max_1d_extreme = ""
        for pattern in max_extreme_patterns:
            # Rechercher dans la section après "Max 1D" et avant "Min 1D" si possible
            max_section = block
            if "Max 1D" in block and "Min 1D" in block:
                max_start = block.find("Max 1D")
                min_start = block.find("Min 1D")
                if max_start < min_start:
                    max_section = block[max_start:min_start]

            extreme_match = re.search(pattern, max_section)
            if extreme_match:
                max_1d_extreme = extreme_match.group(1)
                break

        # Pour les forex où il n'y a pas d'event spécifié, estimer une valeur
        if ticker.startswith("6") and not max_1d_event and max_1d:
            try:
                max_1d_float = float(max_1d)
                max_1d_event = str(round(max_1d_float * 1.005, 6))  # +0.5% arrondi à 6 décimales
            except ValueError:
                pass  # Si conversion impossible, laisser vide

        if not max_1d_event:
            missing_elements.append("Max_1D_event")

        # Extraire Min 1D
        min_1d_patterns = [
            r'Min\s+1D\s+(\d+\.?\d*)',
            r'Min\s+1D\s*(\d+\.?\d*)',  # Sans espace après 1D
            # Patterns en anglais
            r'Min\s+1D\s+(\d+\.?\d*)',
            r'Minimum\s+1D\s+(\d+\.?\d*)'
        ]
        min_1d = ""
        for pattern in min_1d_patterns:
            min_1d_match = re.search(pattern, block)
            if min_1d_match:
                min_1d = min_1d_match.group(1)
                break
        if not min_1d:
            missing_elements.append("Min_1D")

        # Extraire Min 1D event avec patterns améliorés (français et anglais)
        min_1d_section = block[block.find("Min 1D"):] if "Min 1D" in block else block
        min_event_patterns = [
            # Patterns en français
            r'en cas d[\'e]?event\s+(\d+\.?\d*)',
            r'en cas d\s*event\s+(\d+\.?\d*)',  # Avec espace entre 'd' et 'event'
            r'\(en cas d[\'e]?event\s+(\d+\.?\d*)',  # Format avec parenthèses
            r'\(?event\s+(\d+\.?\d*)',  # Format "event X.XXX" ou "(event X.XXX"
            r'\(event\s+(\d+\.?\d*)\)',  # Format "(event X.XXX)"
            # Patterns en anglais
            r'in case of event\s+(\d+\.?\d*)',
            r'\(in case of event\s+(\d+\.?\d*)',
            r'event\s+(\d+\.?\d*)'
        ]
        min_1d_event = ""
        for pattern in min_event_patterns:
            min_1d_event_match = re.search(pattern, min_1d_section)
            if min_1d_event_match:
                min_1d_event = min_1d_event_match.group(1)
                break

        # Extraire Min 1D extreme
        min_extreme_patterns = [
            r'extreme\s+(\d+\.?\d*)',
            r'\(?extreme\s+(\d+\.?\d*)'  # Format avec parenthèse optionnelle
        ]
        min_1d_extreme = ""
        for pattern in min_extreme_patterns:
            extreme_match = re.search(pattern, min_1d_section)
            if extreme_match:
                min_1d_extreme = extreme_match.group(1)
                break

        # Pour les forex où il n'y a pas d'event spécifié
        if ticker.startswith("6") and not min_1d_event and min_1d:
            try:
                min_1d_float = float(min_1d)
                min_1d_event = str(round(min_1d_float * 0.995, 6))  # -0.5% arrondi à 6 décimales
            except ValueError:
                pass  # Si conversion impossible, laisser vide

        if not min_1d_event:
            missing_elements.append("Min_1D_event")

        # Extraire Prise de contrôle acheteurs - patterns améliorés (français et anglais)
        control_achat_patterns = [
            # Patterns en français
            r'[Pp]rise de contrôle acheteurs\s+(\d+\.?\d*)',
            r'[Aa]cheteurs prise de contrôle\s+(\d+\.?\d*)',
            r'[Aa]cheteurs.*contrôle\s+(\d+\.?\d*)',
            r'contrôle.*[Aa]cheteurs\s+(\d+\.?\d*)',
            r'[Pp]rise.*[Aa]cheteurs\s+(\d+\.?\d*)',  # Pattern plus souple
            r'[Aa]cheteurs\s+(\d+\.?\d*)',  # Pattern très simple: juste "acheteurs + nombre"
            # Patterns en anglais
            r'[Bb]uyers control\s+(\d+\.?\d*)',
            r'[Bb]uyers take control\s+(\d+\.?\d*)',
            r'[Bb]uyer control\s+(\d+\.?\d*)',
            r'[Bb]uyer take control\s+(\d+\.?\d*)',
            r'[Bb]uyers.*control\s+(\d+\.?\d*)',
            r'control.*[Bb]uyers\s+(\d+\.?\d*)'
        ]
        control_achat = ""
        for pattern in control_achat_patterns:
            match = re.search(pattern, block)
            if match:
                control_achat = match.group(1)
                break

        # Si toujours pas trouvé, essayer une recherche par ligne avec des mots-clés pour acheteurs
        if not control_achat:
            acheteurs_keywords = ['acheteurs', 'buyers', 'buyer']
            for line in block.split('\n'):
                if any(keyword.lower() in line.lower() for keyword in acheteurs_keywords):
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        control_achat = numbers[-1]  # Prendre le dernier nombre de la ligne
                        break

        if not control_achat:
            missing_elements.append("Control_Acheteurs")

        # Extraire Prise de contrôle vendeurs - patterns améliorés (français et anglais)
        control_vente_patterns = [
            # Patterns en français
            r'[Pp]rise de contrôle vendeurs\s+(\d+\.?\d*)',
            r'[Vv]endeurs prise de contrôle\s+(\d+\.?\d*)',
            r'[Vv]endeurs.*contrôle\s+(\d+\.?\d*)',
            r'contrôle.*[Vv]endeurs\s+(\d+\.?\d*)',
            r'[Pp]rise.*[Vv]endeurs\s+(\d+\.?\d*)',  # Pattern plus souple
            r'[Vv]endeurs\s+(\d+\.?\d*)',  # Pattern très simple: juste "vendeurs + nombre"
            # Patterns en anglais
            r'[Ss]ellers control\s+(\d+\.?\d*)',
            r'[Ss]ellers take control\s+(\d+\.?\d*)',
            r'[Ss]eller control\s+(\d+\.?\d*)',
            r'[Ss]eller take control\s+(\d+\.?\d*)',
            r'[Ss]ellers.*control\s+(\d+\.?\d*)',
            r'control.*[Ss]ellers\s+(\d+\.?\d*)'
        ]
        control_vente = ""
        for pattern in control_vente_patterns:
            match = re.search(pattern, block)
            if match:
                control_vente = match.group(1)
                break

        # Si toujours pas trouvé, essayer une recherche par ligne avec des mots-clés pour vendeurs
        if not control_vente:
            vendeurs_keywords = ['vendeurs', 'sellers', 'seller']
            for line in block.split('\n'):
                if any(keyword.lower() in line.lower() for keyword in vendeurs_keywords):
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        control_vente = numbers[-1]  # Prendre le dernier nombre de la ligne
                        break

        if not control_vente:
            missing_elements.append("Control_Vendeurs")

        # Extraire Put sup - support amélioré pour multiples valeurs (français et anglais)
        put_sup_patterns = [
            # Patterns en français
            r'[Pp]ut sup\.?\s+0dte\s+([\d\., ]+)',
            r'[Pp]ut sup\.?\s+([\d\., ]+)',
            r'[Pp]ut sup\s+([\d\., ]+)',  # Variante sans point
            # Patterns en anglais
            r'[Pp]ut support\.?\s+0dte\s+([\d\., ]+)',
            r'[Pp]ut support\.?\s+([\d\., ]+)',
            r'[Pp]ut support\s+([\d\., ]+)',
            r'[Pp]ut sup\s+([\d\., ]+)'
        ]

        # Détection de Put sup. (ou support)
        put_sup_line = ""
        for line in block.split('\n'):
            if re.search(r'[Pp]ut sup|[Pp]ut support', line):
                put_sup_line = line
                break

        put_sup_main, put_sup_all = "", ""
        if put_sup_line:
            # Exclut les zéros (le « 0 » de 0dte)
            nums = [v for v in re.findall(r'\d+\.?\d*', put_sup_line) if float(v) != 0]
            if nums:
                put_sup_main = nums[0]
                put_sup_all = ", ".join(nums)
        else:
            # Fallback pattern
            for pattern in put_sup_patterns:
                m = re.search(pattern, block)
                if m:
                    nums = [v for v in re.findall(r'\d+\.?\d*', m.group(1)) if float(v) != 0]
                    if nums:
                        put_sup_main = nums[0]
                        put_sup_all = ", ".join(nums)
                    break

        if not put_sup_main:
            missing_elements.append("Put_Sup")

        # Extraire Call res - support amélioré pour multiples valeurs (français et anglais)
        call_res_patterns = [
            # Patterns en français
            r'[Cc]all res\.?\s+0dte\s+([\d\., ]+)',
            r'[Cc]all res\.?\s+([\d\., ]+)',
            r'[Cc]all res\s+([\d\., ]+)',  # Variante sans point
            # Patterns en anglais
            r'[Cc]all resistance\.?\s+0dte\s+([\d\., ]+)',
            r'[Cc]all resistance\.?\s+([\d\., ]+)',
            r'[Cc]all resistance\s+([\d\., ]+)',
            r'[Cc]all res\s+([\d\., ]+)'
        ]

        # Détection de Call res. (ou resistance)
        call_res_line = ""
        for line in block.split('\n'):
            if re.search(r'[Cc]all res|[Cc]all resistance', line):
                call_res_line = line
                break

        call_res_main, call_res_all = "", ""
        if call_res_line:
            nums = [v for v in re.findall(r'\d+\.?\d*', call_res_line) if float(v) != 0]
            if nums:
                call_res_main = nums[0]
                call_res_all = ", ".join(nums)
        else:
            for pattern in call_res_patterns:
                m = re.search(pattern, block)
                if m:
                    nums = [v for v in re.findall(r'\d+\.?\d*', m.group(1)) if float(v) != 0]
                    if nums:
                        call_res_main = nums[0]
                        call_res_all = ", ".join(nums)
                    break

        if not call_res_main:
            missing_elements.append("Call_Res")

        # Extraire les niveaux "all" pour Put sup - amélioration pour détecter les valeurs entre parenthèses
        # Patterns en français et anglais
        all_patterns = [
            r'\(all\s+([\d\.,\s]+)\)',  # Pattern pour (all X, Y, Z)
            r'\(tous\s+([\d\.,\s]+)\)',  # Pattern français (tous X, Y, Z)
            r'put all\s+([\d\.,\s]+)'  # Pattern "put all X, Y, Z"
        ]

        for pattern in all_patterns:
            all_match = re.search(pattern, block, re.IGNORECASE)
            if all_match:
                all_values_text = all_match.group(1)
                all_values = re.findall(r'\d+\.?\d*', all_values_text)
                if all_values:
                    if "Put" in block[block.find(all_match.group(0)) - 30:block.find(all_match.group(0))]:
                        # C'est pour les puts
                        if put_sup_all:
                            put_sup_all += ", " + ", ".join(all_values)
                        else:
                            put_sup_all = ", ".join(all_values)
                    elif "Call" in block[block.find(all_match.group(0)) - 30:block.find(all_match.group(0))]:
                        # C'est pour les calls
                        if call_res_all:
                            call_res_all += ", " + ", ".join(all_values)
                        else:
                            call_res_all = ", ".join(all_values)

        # Extraire explicitement put all et call all (français et anglais)
        put_all_patterns = [
            # Patterns en français
            r'put all\s+([\d\., ]+)',
            r'[Pp]ut.*\(all\s+([\d\., ]+)\)',  # Format "(all X.XX)" pour puts
            # Patterns en anglais
            r'put all\s+([\d\., ]+)',
            r'[Pp]ut.*\(all\s+([\d\., ]+)\)'
        ]

        for pattern in put_all_patterns:
            put_all_match = re.search(pattern, block, re.IGNORECASE)
            if put_all_match:
                all_values = re.findall(r'\d+\.?\d*', put_all_match.group(1))
                if all_values:
                    if put_sup_all:
                        put_sup_all += ", " + ", ".join(all_values)
                    else:
                        put_sup_all = ", ".join(all_values)
                break

        # Extraire les valeurs hedgies
        hedgies_patterns = [
            r'hedgies\s+([\d\., ]+)',
            r'hedge\s+([\d\., ]+)'
        ]

        for pattern in hedgies_patterns:
            hedgies_match = re.search(pattern, block, re.IGNORECASE)
            if hedgies_match:
                hedgies_values = re.findall(r'\d+\.?\d*', hedgies_match.group(1))
                if hedgies_values:
                    if put_sup_all:
                        put_sup_all += ", Hedgies: " + ", ".join(hedgies_values)
                    else:
                        put_sup_all = "Hedgies: " + ", ".join(hedgies_values)

        # Extraire explicitement call all (français et anglais)
        call_all_patterns = [
            # Patterns en français
            r'call all\s+([\d\., ]+)',
            r'[Cc]all.*\(all\s+([\d\., ]+)\)',  # Format "(all X.XX)" pour calls
            # Patterns en anglais
            r'call all\s+([\d\., ]+)',
            r'[Cc]all.*\(all\s+([\d\., ]+)\)'
        ]

        for pattern in call_all_patterns:
            call_all_match = re.search(pattern, block, re.IGNORECASE)
            if call_all_match:
                all_values = re.findall(r'\d+\.?\d*', call_all_match.group(1))
                if all_values:
                    if call_res_all:
                        call_res_all += ", " + ", ".join(all_values)
                    else:
                        call_res_all = ", ".join(all_values)
                break

        # Enregistrer les erreurs si des éléments sont manquants
        if missing_elements:
            errors.append(
                f"Date {current_date}, Ticker {ticker}: Éléments manquants: {', '.join(missing_elements)}")
        else:
            print(f"Toutes les données ont été trouvées pour {ticker}")

        # Ajouter les données à la liste
        all_data.append({
            'Date': current_date,
            'Ticker': ticker,  # Utiliser le ticker tel quel
            'Base_Ticker': base_ticker,
            'Gamma': gamma,
            'Max_1D': max_1d,
            'Max_1D_event': max_1d_event,
            'Max_1D_extreme': max_1d_extreme,  # Nouveau champ pour les valeurs extreme
            'Min_1D': min_1d,
            'Min_1D_event': min_1d_event,
            'Min_1D_extreme': min_1d_extreme,  # Nouveau champ pour les valeurs extreme
            'Control_Acheteurs': control_achat,
            'Control_Vendeurs': control_vente,
            'Put_Sup_Main': put_sup_main,
            'Put_Sup_All': put_sup_all,
            'Call_Res_Main': call_res_main,
            'Call_Res_All': call_res_all
        })

    return all_data, errors


def save_to_csv(data, output_file):
    """
    Sauvegarde les données extraites dans un fichier CSV.
    """
    if not data:
        print("Aucune donnée à sauvegarder.")
        return False

    # Ajouter les nouveaux champs pour les valeurs extreme
    fieldnames = [
        'Date', 'Ticker', 'Base_Ticker', 'Gamma', 'Max_1D', 'Max_1D_event', 'Max_1D_extreme',
        'Min_1D', 'Min_1D_event', 'Min_1D_extreme', 'Control_Acheteurs', 'Control_Vendeurs',
        'Put_Sup_Main', 'Put_Sup_All', 'Call_Res_Main', 'Call_Res_All'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(data)

    print(f"Données sauvegardées dans {output_file}")
    return True


def process_file(input_file, output_dir=None):
    """
    Traite un fichier brouillon et génère le fichier CSV correspondant.
    """
    # Extraire la date du nom de fichier
    date_match = re.search(r'raw_level_(\d{8})\.txt', os.path.basename(input_file))
    if date_match:
        date_str = date_match.group(1)
        expected_date = f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
        output_file = f"SC_level_{date_str}.csv"
    else:
        # Utiliser la date actuelle si la date n'est pas dans le nom du fichier
        current_date = datetime.now().strftime('%d%m%Y')
        expected_date = None
        output_file = f"SC_level_{current_date}.csv"

    # Définir le répertoire de sortie
    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    output_path = os.path.join(output_dir, output_file)

    # Analyser le fichier et sauvegarder les données
    data, errors = parse_raw_file(input_file, expected_date)

    print(f"Données extraites: {len(data)} entrées")
    if len(data) > 0:
        print("Premier ticker trouvé:", data[0]["Ticker"] if "Ticker" in data[0] else "Inconnu")

    # Afficher les erreurs s'il y en a
    if errors:
        print(f"\nATTENTION - Éléments manquants dans le fichier {os.path.basename(input_file)}:")
        for error in errors:
            print(f"  - {error}")

        # Créer un fichier de log d'erreurs
        error_log_file = os.path.join(output_dir, f"errors_{os.path.basename(input_file).replace('.txt', '.log')}")
        with open(error_log_file, 'w', encoding='utf-8') as error_file:
            error_file.write(f"Erreurs détectées lors de l'analyse du fichier {input_file}:\n\n")
            for error in errors:
                error_file.write(f"- {error}\n")

        print(f"Les erreurs ont été enregistrées dans {error_log_file}")

        # Demander à l'utilisateur s'il souhaite continuer malgré les erreurs
        choice = input("\nDes éléments sont manquants. Souhaitez-vous tout de même générer le fichier CSV ? (O/N) : ")
        if choice.upper() != 'O':
            print("Génération du fichier CSV annulée.")
            return None

    # Sauvegarde des données et vérification du succès
    if save_to_csv(data, output_path):
        return output_path
    else:
        print("Aucun fichier CSV n'a été généré en raison de l'absence de données.")
        return None


if __name__ == "__main__":
    # Chemin du répertoire contenant les fichiers brouillons
    base_dir = r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\discord_derivative"
    print(f"Répertoire de travail: {base_dir}")

    # Trouver tous les fichiers raw_level_*.txt
    raw_files = [f for f in os.listdir(base_dir) if f.startswith("raw_level_") and f.endswith(".txt")]

    if not raw_files:
        print("Aucun fichier raw_level_*.txt trouvé dans le répertoire.")
        exit()


    # Trier les fichiers par date dans le nom (du plus récent au plus ancien)
    def extract_date(filename):
        match = re.search(r'raw_level_(\d{2})(\d{2})(\d{4})\.txt', filename)
        if match:
            jour, mois, annee = match.groups()
            try:
                return datetime(int(annee), int(mois), int(jour))
            except ValueError:
                return datetime(1900, 1, 1)  # Date par défaut si invalide
        return datetime(1900, 1, 1)  # Date par défaut si format incorrect


    raw_files.sort(key=extract_date, reverse=True)

    # Proposer le fichier le plus récent
    latest_file = raw_files[0]
    date_match = re.search(r'raw_level_(\d{2})(\d{2})(\d{4})\.txt', latest_file)
    if date_match:
        jour, mois, annee = date_match.groups()
        date_fichier = f"{jour}/{mois}/{annee}"
        print(f"\nFichier le plus récent détecté: {latest_file} (date: {date_fichier})")
    else:
        print(f"\nFichier le plus récent détecté: {latest_file}")

    # Demander confirmation ou saisie d'un nouveau nom de fichier
    response = input("\nAppuyez sur Entrée pour utiliser ce fichier, ou saisissez un autre nom de fichier: ")

    if response.strip():
        # L'utilisateur a saisi un nom de fichier
        file_name = response.strip()
        if not file_name.endswith(".txt"):
            file_name += ".txt"
        if not file_name.startswith("raw_level_"):
            file_name = "raw_level_" + file_name
    else:
        # L'utilisateur a accepté le fichier proposé
        file_name = latest_file

    # Vérifier si le fichier existe
    input_file = os.path.join(base_dir, file_name)
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {file_name} n'existe pas dans le répertoire {base_dir}")
        exit()

    # Traiter le fichier
    output_path = process_file(input_file)
    if output_path:
        print(f"Traitement terminé. Fichier CSV généré : {output_path}")