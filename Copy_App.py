
# Streamlit App Setup
st.set_page_config(page_title="Bird Sound Classifier", layout="wide")

# Sidebar für Navigation
pages = ["🏠 Welcome_Page", "📂 Audio-Upload", "📊 Spectrogram", "🔍 Analysis"]
page = st.sidebar.radio("Navigation", pages)


# Welcome_page
if page == "🏠 Welcome_Page":
    st.title("🎶🐦 Is this bird endangered?")
    st.write("This app helps you to classify bird sounds using a pre-trained model.")
    if st.button("Let's go! 🎵"):
        st.session_state["page"] = "📂 Audio-Upload"
        st.rerun()

# Audio-Upload
elif page == "📂 Audio-Upload":
    st.title("📂 Upload audio file")
    uploaded_file = st.file_uploader("Upload an Audio Data", type=["wav", "mp3", "ogg"])

    if uploaded_file:
        file_path = "temp_audio.ogg"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(file_path, format="audio/ogg")
        st.write("✅ file saved successfully!")
        st.session_state["file_path"] = file_path
        
        if st.button("To the Spectrogram"):
            st.session_state["page"] = "📊 Spectrogram"
            st.rerun()

# spectrogram 
elif page == "📊 Spectrogram":
    st.title("📊 Mel-Spectrogram")
    
    if "file_path" in st.session_state:
        file_path = st.session_state["file_path"]
        y, sr = librosa.load(file_path, sr=22050)
        
        if len(y) == 0:
            st.error("❌ Empty audio file, please upload a valid file!")
            st.stop()

        st.write(f"Maximale Amplitude: {np.max(y)}")
        st.write(f"Looks like a valid audio file! 🎉")

        # Mel-Spektrogramm berechnen
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S + 1e-6, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        st.pyplot(fig)

        st.session_state["spectrogram"] = S_dB

        if st.button("🔍 To Analyis"):
            st.session_state["page"] = "🔍 Analyse"
            st.rerun()
    else:
        st.warning("Please upload a file first!")

# analysis
elif page == "🔍 Analysis":
    st.title("🔍 AI Analysis on it's Endangered Status")

    if "Spectrogram" in st.session_state:
        st.success("📊 Spectrogram loaded, ready to analyze!")

        # Modell-Integration (wenn später verfügbar)
        MODEL_PATH = "bird_model.pth"  # UPDATE, wenn Modell vorhanden
        try:
            model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.eval()  # Modell auf Inferenz setzen

            # Spektrogramm für CNN vorbereiten
            S = np.expand_dims(st.session_state["spectrogram"], axis=0)
            S_tensor = torch.tensor(S).unsqueeze(0)  # (1, Height, Width)

            # Modellvorhersage
            prediction = model(S_tensor)
            st.write("📢 prediction:", prediction)  # TODO: Ausgabe formatieren

        except FileNotFoundError:
            st.warning("❌ model not found, please upload a model.")
    else:
        st.warning("No Spectrogram found, please upload a file first!")