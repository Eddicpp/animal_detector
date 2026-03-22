# Esportiamo le classi principali così sono accessibili direttamente da 'src'
from .detector import AnimalDetector
from .processor import ImageProcessor
from .utils import get_animal_info, calculate_session_score

# Possiamo anche definire una versione del nostro software
__version__ = "1.0.0"