import streamlit as st
import cv2
from src import AnimalDetector, ImageProcessor, get_animal_info

# Configurazione della pagina
st.set_page_config(page_title="EcoTracker AI", page_icon="🐾", layout="wide")

st.title("🐾 EcoTracker: Rilevamento Animali Pro")
st.markdown("---")

# 1. Inizializzazione (Carichiamo il cervello e le mani)
@st.cache_resource # Questo evita di ricaricare il modello ogni volta che clicchi qualcosa
def load_assets():
    detector = AnimalDetector("models/animali_v1.pt")
    processor = ImageProcessor()
    return detector, processor

detector, processor = load_assets()

# Modifica nel file app.py

# ... (codice precedente di inizializzazione) ...

# 2. Barra Laterale (Modificata per caricamento multiplo)
st.sidebar.header("Impostazioni")
conf_threshold = st.sidebar.slider("Soglia di Confidenza", 0.0, 1.0, 0.45)

# Aggiungiamo accept_multiple_files=True
uploaded_files = st.sidebar.file_uploader(
    "Carica una o più foto", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

# 3. Logica Principale (Aggiornata con un ciclo for)
if uploaded_files:
    st.write(f"📂 Elaborazione di **{len(uploaded_files)}** immagini...")
    
    # Creiamo un riassunto globale per la sessione
    global_counts = {}

    for uploaded_file in uploaded_files:
        # Trasforma il file in immagine
        image = processor.get_image_from_upload(uploaded_file)
        
        # Esegui l'IA
        results = detector.predict(image, confidence=conf_threshold, iou=0.7)
        detections = detector.get_formatted_results(results)
        
        # Creiamo un'area dedicata per ogni immagine nel batch
        with st.expander(f"Risultato per: {uploaded_file.name}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                annotated_img = processor.draw_custom_boxes(image, detections)
                st.image(annotated_img, use_container_width=True)
            
            with col2:
                if detections:
                    img_counts = detector.count_animals(detections)
                    st.write("**Animali trovati:**")
                    for species, count in img_counts.items():
                        st.write(f"- {count}x {species}")
                        # Aggiorna il conteggio globale della sessione
                        global_counts[species] = global_counts.get(species, 0) + count
                else:
                    st.info("Nessun rilevamento in questa foto.")

    # --- RIASSUNTO FINALE DEL BATCH ---
    st.markdown("---")
    st.header("📊 Riassunto Sessione")
    if global_counts:
        cols = st.columns(len(global_counts))
        for i, (species, total) in enumerate(global_counts.items()):
            cols[i].metric(label=species, value=total)
    else:
        st.write("Nessun animale rilevato nell'intero set.")

else:
    st.info("👈 Seleziona una serie di foto dalla barra laterale per il test di massa.")