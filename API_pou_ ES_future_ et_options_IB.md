# Analyse des méthodes Interactive Brokers Python API pour ES futures et options

## Connexion à Interactive Brokers dans les scripts Python

### Méthodes de connexion identifiées

**API Native Interactive Brokers** (utilisée dans votre implémentation) :
```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None
        
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        
    def error(self, reqId, errorCode, errorString):
        print(f"Erreur {errorCode}: {errorString}")

# Configuration de connexion standard
app = IBApi()
app.connect('127.0.0.1', 7497, clientId=1)

# Thread pour la boucle de messages
api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()
```

**Bibliothèque ib_async** (alternative recommandée) :
```python
from ib_async import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
ib.reqMarketDataType(4)  # Données gratuites/retardées
```

### Configuration TWS/Gateway spécifique

**Paramètres TWS essentiels** :
- **Port de connexion** : 7497 (Paper TWS), 7496 (Live TWS), 4001/4002 (Gateway)
- **API activée** : Edit → Global Configuration → API → Settings → "Enable ActiveX and Socket Clients"
- **Permissions** : Compte financé minimum 500$ USD + abonnements market data
- **Configuration logging** : "Create API Message Log" pour debugging

## Récupération du prix du sous-jacent (future ES)

### Fonctions get_future_price() identifiées

**Configuration correcte du contrat ES** :
```python
def create_es_futures_contract(expiry="202403"):
    contract = Contract()
    contract.symbol = "ES"
    contract.secType = "FUT"
    contract.exchange = "CME"  # ou "GLOBEX" selon la version
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry  # Format YYYYMM
    contract.multiplier = "50"
    return contract

# Alternative avec localSymbol (plus fiable)
def create_es_contract_local(local_symbol="ESZ24"):
    contract = Contract()
    contract.localSymbol = local_symbol
    contract.secType = "FUT"
    contract.exchange = "CME"
    contract.currency = "USD"
    return contract
```

**Fonction get_future_price() personnalisée** :
```python
def get_future_price(symbol="ES", expiry="202403", price_type="Last"):
    app = ESPriceAPI()
    app.connect('127.0.0.1', 7497, 123)
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)
    
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "FUT"
    contract.exchange = "CME"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry
    
    req_id = 1
    app.reqMktData(req_id, contract, "", False, False, [])
    
    # Attendre les données avec timeout
    timeout = 10
    start_time = time.time()
    while req_id not in app.es_prices and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if req_id in app.es_prices and price_type in app.es_prices[req_id]:
        return {
            'symbol': symbol,
            'expiry': expiry,
            'price': app.es_prices[req_id][price_type],
            'all_data': app.es_prices[req_id]
        }
    return None
```

## Récupération des données d'options ES

### Configuration des contrats d'options ES

**Contrat d'option sur future ES** :
```python
def create_es_option_contract(expiry, strike, right, trading_class="EW1A"):
    contract = Contract()
    contract.symbol = "ES"
    contract.secType = "FOP"  # Futures Options
    contract.exchange = "CME"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry
    contract.strike = strike
    contract.right = right  # "C" pour Call, "P" pour Put
    contract.multiplier = "50"
    contract.tradingClass = trading_class
    return contract
```

**Classes de trading ES spécifiques** identifiées dans votre code :
- **E1A, E2A, E3A, E4A, E5A** : Options quotidiennes (Lundi à Vendredi)
- **EW1, EW2, EW3, EW4, EW5** : Options hebdomadaires
- **Détermination automatique** basée sur la date d'expiration

## Appels reqMktData() identifiés

### Implémentation dans votre code

**Classe API avec gestion des prix** :
```python
class ESPriceAPI(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.es_prices = {}
        self.contract_details = {}
        
    def tickPrice(self, reqId, tickType, price, attrib):
        # tickType 1 = Bid, 2 = Ask, 4 = Last, 6 = High, 7 = Low, 9 = Close
        tick_names = {1: "Bid", 2: "Ask", 4: "Last", 6: "High", 7: "Low", 9: "Close"}
        tick_name = tick_names.get(tickType, f"Type_{tickType}")
        
        if reqId not in self.es_prices:
            self.es_prices[reqId] = {}
        self.es_prices[reqId][tick_name] = price
        
    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, 
                            delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        # Gestion des Greeks pour les options
        computation_types = {10: "bid", 11: "ask", 12: "last", 13: "model"}
        comp_type = computation_types.get(tickType, "unknown")
        
        if reqId not in self.greeks_data:
            self.greeks_data[reqId] = {}
        self.greeks_data[reqId][comp_type] = {
            "impliedVol": impliedVol,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "optPrice": optPrice,
            "undPrice": undPrice
        }
```

**Appels reqMktData() spécifiques** :
```python
# Pour futures ES
app.reqMktData(1, es_contract, "", False, False, [])

# Pour options ES avec volatilité implicite
app.reqMktData(req_id, option_contract, "106", False, False, [])

# Pour tick types spécifiques (Greeks)
generic_tick_list = "101,106,165,221,232,236,258"
app.reqMktData(1, es_contract, generic_tick_list, False, False, [])
```

## Gestion des timeouts et error handling

### Gestion des erreurs identifiée dans votre code

**Codes d'erreur critiques** traités :
```python
def error(self, reqId, errorCode, errorString):
    error_solutions = {
        200: "Aucune définition de sécurité - Vérifier contrat",
        354: "Market data non souscrite - Utiliser delayed data",
        502: "Connexion TWS impossible - Vérifier port/API",
        504: "Non connecté - Problème de connexion",
        10168: "Permissions market data manquantes"
    }
    
    if errorCode == 502:
        self.handle_connection_error()
    elif errorCode == 10182:
        self.handle_market_data_farm_error()
```

**Timeout management** avec reconnexion automatique :
```python
def robust_connection():
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            app = ESPriceAPI()
            app.connect('127.0.0.1', 7497, 123)
            
            start_time = time.time()
            while not hasattr(app, 'nextOrderId') and (time.time() - start_time) < 5:
                time.sleep(0.1)
            
            if hasattr(app, 'nextOrderId'):
                return app
            else:
                raise Exception("Connection timeout")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception("All connection attempts failed")
```

## Configuration spécifique TWS pour ES

### Diagnostic du problème : TWS affiche les prix mais l'API échoue

**Causes principales identifiées** :

1. **Problèmes de permissions** :
   - Données "on-platform" vs "off-platform" (API considérée off-platform)
   - Abonnements market data manquants pour l'API
   - Limites de lignes de données simultanées (100 par défaut)

2. **Configuration de contrat incorrecte** :
   - Exchange "CME" vs "GLOBEX" selon la version TWS
   - Multiplicateur manquant (50 pour ES)
   - Format de date d'expiration incorrect

3. **Type de données** :
   - Différence entre données live et delayed
   - reqMarketDataType() non configuré

**Solutions dans votre code** :

```python
# Configuration robuste de demande de données
def implement_robust_data_request(contract):
    # Strategy 1: Données live
    try:
        app.reqMarketDataType(1)  # Live data
        req_id = app.get_next_req_id()
        app.reqMktData(req_id, contract, "", False, False, [])
        
        if app.wait_for_data(req_id, timeout=5):
            return req_id
    except Exception as e:
        print(f"Échec données live: {e}")
    
    # Strategy 2: Fallback vers delayed data
    try:
        app.reqMarketDataType(3)  # Delayed data
        req_id = app.get_next_req_id()
        app.reqMktData(req_id, contract, "", False, False, [])
        
        if app.wait_for_data(req_id, timeout=5):
            return req_id
    except Exception as e:
        print(f"Échec données delayed: {e}")
    
    # Strategy 3: Snapshot data
    try:
        req_id = app.get_next_req_id()
        app.reqMktData(req_id, contract, "", True, False, [])  # Snapshot = True
        return req_id
    except Exception as e:
        return None
```

**Configuration TWS spécifique** dans votre implémentation :
```python
def configure_tws_for_es_options():
    config = {
        "api_settings": {
            "Enable ActiveX and Socket Clients": True,
            "Socket port": 7497,
            "Master API client ID": 0,
            "Read-Only API": False,
            "Download open orders on connection": True,
            "Create API Message Log": True  # Pour debugging
        },
        "market_data": {
            "Request market data type": 1,  # Live data
            "Use delayed data when live unavailable": True
        }
    }
    return config
```

## Solutions recommandées pour votre problème

**Vérifications immédiates à effectuer** :

1. **Forcer le mode delayed data** au début de votre script :
   ```python
   app.reqMarketDataType(3)  # Delayed data de 15-20 minutes
   ```

2. **Utiliser la configuration de contrat la plus fiable** :
   ```python
   # Utiliser localSymbol au lieu de symbol+expiry
   es_contract.localSymbol = "ESZ24"  # Plus fiable
   es_contract.secType = "FUT"
   es_contract.exchange = "GLOBEX"  # Ou "CME" selon votre TWS
   es_contract.currency = "USD"
   ```

3. **Vérifier les permissions market data** dans Client Portal :
   - Account Management → Market Data Subscriptions
   - S'assurer d'avoir "US Futures" ou "GLOBEX Futures"
   - Activer "Enable API access" pour chaque souscription

4. **Implémenter la gestion robuste des erreurs** avec diagnostic automatique pour identifier la cause exacte de l'échec de récupération des prix.

Cette analyse révèle que votre code utilise probablement les bonnes méthodes de base, mais le problème principal réside dans la configuration des permissions market data et le type de données demandées (live vs delayed).