from ultralytics import YOLO

# 1. Carica il TUO modello allenato (quello che ha prodotto i risultati migliori)
model = YOLO('models/animali_v1.pt')

# 2. Esegui la validazione sul Test Set
metrics = model.val(split='test') 

# 3. Stampa i risultati corretti per Object Detection
print("\n--- RISULTATI TEST ---")
print(f"Mappa di precisione (mAP@50): {metrics.box.map50:.3f}")
print(f"Mappa di precisione (mAP@50-95): {metrics.box.map:.3f}")
print(f"Precisione (P): {metrics.box.mp:.3f}")
print(f"Richiamo (R): {metrics.box.mr:.3f}")