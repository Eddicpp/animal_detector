from ultralytics import YOLO

def main():
    # 1. Carica il modello pre-addestrato (partiamo dalla versione "nano" per velocità, 
    # ma puoi usare "yolov8s.pt" se vuoi un modello leggermente più preciso e pesante)
    model = YOLO("models/animali_v1.pt") 

    print("🚀 Inizio dell'addestramento del modello...")

    # 2. Avvia il training con i parametri ottimizzati
    model.train(
        data="data/data.yaml", 
        epochs=40,              # Ne bastano 40: è un perfezionamento, non un trasloco
        imgsz=640,
        batch=16,
        lr0=0.001,              # Learning rate BASSO: non vogliamo che "dimentichi" il passato
        patience=15,            # Se dopo 15 epoche non migliora i falsi positivi, si ferma
        
        # Manteniamo le augmentation per rendere i nuovi sfondi ancora più vari
        mosaic=1.0,             
        mixup=0.1,
        
        project="runs/detect",
        name="eco_tracker_v2_final", # Nuovo nome per non sovrascrivere
        exist_ok=True
    )

    print("✅ Addestramento completato! Controlla la cartella runs/detect/eco_tracker_v1/ per i risultati.")

# Questo blocco if è fondamentale per evitare crash se avvii il codice su Windows
if __name__ == '__main__':
    main()