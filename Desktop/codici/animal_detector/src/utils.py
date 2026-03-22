# src/utils.py

def format_confidence(conf_value):
    """Trasforma 0.987 in 99%"""
    return f"{round(conf_value * 100)}%"

def get_conservation_status(species):
    """Associa a ogni specie il suo stato di conservazione"""
    status_map = {
        "Lion": "Vulnerabile (VU)",
        "Elephant": "In Pericolo (EN)",
        "Zebra": "Rischio Minimo (LC)"
    }
    return status_map.get(species, "Dato non disponibile")

def get_animal_fact(species):
    """Restituisce un fatto curioso per l'angolo educativo"""
    facts = {
        "Lion": "I leoni sono gli unici felini che vivono in branchi sociali.",
        "Elephant": "Le zanne degli elefanti sono in realtà denti incisivi.",
        "Zebra": "Le strisce delle zebre sono uniche come le impronte digitali."
    }
    return facts.get(species, "Sapevi che ogni animale è fondamentale per l'ecosistema?")

# --- QUESTE SONO LE FUNZIONI CHE MANCAVANO ---

def get_animal_info(species):
    """
    Funzione 'Ponte' richiesta da app.py.
    Raccoglie le info dalle altre funzioni e le restituisce in un colpo solo.
    """
    return {
        "status": get_conservation_status(species),
        "fact": get_animal_fact(species)
    }

def calculate_session_score(stats):
    """Calcola un punteggio basato sul numero di animali trovati"""
    return sum(stats.values()) * 10