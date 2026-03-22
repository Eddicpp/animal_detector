from ultralytics import YOLO
import numpy as np

class AnimalDetector:
    """
    Classe responsabile della gestione del modello YOLOv8 per il 
    rilevamento e la classificazione di animali.
    """
    
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """
        Inizializza il modello caricando i pesi.
        :param model_path: Percorso del file .pt del modello.
        """
        # Carichiamo il modello (Nano di default per la velocità su laptop)
        self.model = YOLO(model_path)
        
    def predict(self, image: np.ndarray, confidence: float = 0.25, iou=0.45):
        """
        Esegue l'inferenza su un'immagine.
        :param image: Immagine in formato numpy array (caricata da OpenCV o Streamlit).
        :param confidence: Soglia minima di confidenza (0.0 - 1.0).
        :return: Oggetto Results di Ultralytics.
        """
        # Task="detect" specifica che vogliamo fare object detection
        results = self.model.predict(
            source=image, 
            conf=confidence, 
            save=False, 
            verbose=False
        )
        return results[0]  # Restituiamo i risultati per la singola immagine

    def get_formatted_results(self, result) -> list:
        """
        Trasforma i risultati grezzi in una lista di dizionari leggibili.
        Utile per creare log, CSV o visualizzazioni custom.
        """
        detections = []
        for box in result.boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()  # Coordinate [x1, y1, x2, y2]
            })
        return detections

    def count_animals(self, detections: list) -> dict:
        """
        Conta quanti esemplari di ogni specie sono stati rilevati.
        """
        counts = {}
        for det in detections:
            species = det["class"]
            counts[species] = counts.get(species, 0) + 1
        return counts