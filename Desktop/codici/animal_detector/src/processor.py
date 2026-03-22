import cv2
import pandas as pd
from datetime import datetime
import numpy as np

class ImageProcessor:
    """
    Gestisce le operazioni di manipolazione immagini e 
    salvataggio dati post-rilevamento.
    """

    @staticmethod
    def draw_custom_boxes(image, detections):
        """
        Disegna bounding box personalizzate. Invece dei classici rettangoli,
        possiamo usare angoli smussati o colori specifici per il nostro brand.
        """
        img_canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Disegna il rettangolo (Verde Eco)
            cv2.rectangle(img_canvas, (x1, y1), (x2, y2), (46, 204, 113), 2)
            
            # Aggiunge l'etichetta con sfondo
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_canvas, (x1, y1 - 20), (x1 + w, y1), (46, 204, 113), -1)
            cv2.putText(img_canvas, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img_canvas

    @staticmethod
    def save_to_csv(detections, output_path="logs/detections_history.csv"):
        """
        Salva i risultati in un log storico con timestamp.
        """
        if not detections:
            return

        new_data = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for det in detections:
            new_data.append({
                "timestamp": timestamp,
                "species": det["class"],
                "confidence": det["confidence"]
            })
        
        df = pd.DataFrame(new_data)
        # Append al file esistente o creane uno nuovo
        try:
            df.to_csv(output_path, mode='a', header=not pd.io.common.file_exists(output_path), index=False)
        except Exception as e:
            print(f"Errore nel salvataggio CSV: {e}")

    @staticmethod
    def get_image_from_upload(uploaded_file):
        """
        Converte un file caricato su Streamlit in un formato utilizzabile da OpenCV.
        """
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)