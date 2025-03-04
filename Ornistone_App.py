import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import sqlite3



# Streamlit App Setup
st.set_page_config(page_title="Bird Sound Classifier", layout="wide")

# Sidebar für Navigation
with st.sidebar:
    st.title("Navigation")
    st.image("images/line_dodo.png", use_container_width=True)
    pages = ["🏠 Welcome_Page", "📂 Audio-Upload", "📊 Spectrogram", "🔍 Analysis", "📝 Metadata Survey"]
    page = st.radio("Go to", pages, index=0)

#########################################################################################################################################
# Welcome_page
if page == "🏠 Welcome_Page":
    st.title("🎶🐦 What is this bird?")
    st.write("Hello fellow Bird Enthusiast! Pleased to have you. Did you ever wander through the forest and heard a perculiar song, one which you cannot categorize? Well, this App helps identifying it! With the help of a preptrained AI model, we can predict the species of the bird. In the current version it is solely able to tell you if it is endangered or not. Further updates are to come 🦜🌳")
    st.write("")
 # image
    st.image('images/Ornithologist.png', width=450)
 #button
    if st.button("Recorded a sound? Click here! 🎵"):  # Fixed emoji
        st.session_state["page"] = "📂 Audio-Upload"
        st.rerun()
 # blank space

    

#########################################################################################################################################
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
        
        if st.button("To Spectrogram 📊"):  # Fixed button text
            st.session_state["page"] = "📊 Spectrogram"
            st.rerun()

#########################################################################################################################################
# Spectrogram
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

        if st.button("To Analysis 🔍"):  # Fixed button text
            st.session_state["page"] = "🔍 Analysis"
            st.rerun()
    else:
        st.warning("Please upload a file first!")

#########################################################################################################################################
# Analysis
elif page == "🔍 Analysis":
    st.title("🔍 AI Analysis on it's Endangered Status")

    if "spectrogram" in st.session_state:  # Fixed key name (Spectrogram → spectrogram)
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


#########################################################################################################################################
# Metadata Survey Page
elif page == "📝 Metadata Survey":
    st.title("📝 Metadata Survey")
    st.write("Please provide additional information about the recording.")

    # Location Input
    location = st.text_input("📍 Location (e.g., city, country, or coordinates):")

    # Weather Conditions Dropdown
    weather_options = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Other"]
    weather = st.selectbox("☁️ Weather Conditions:", weather_options)

    # Time of Recording (Streamlit's Native Time Input)
    st.write("⏰ Time of Recording:")
    time = st.time_input("Select Time", value=None)  # No default time

    # Further Notes
    notes = st.text_area("📝 Further Notes (e.g., bird behavior):")

    # Save Metadata to Session State
    if st.button("Save Metadata"):
        st.session_state["metadata"] = {
            "location": location,
            "weather": weather,
            "time": str(time),  # Convert time to string
            "notes": notes
        }
        st.success("✅ Metadata saved successfully!")


    conn = sqlite3.connect("metadata.db")

    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metadata
             (location TEXT, weather TEXT, time TEXT, notes TEXT)''')
    conn.commit()
    conn.close()
    # Display Saved Metadata
    if "metadata" in st.session_state:
        st.write("### Saved Metadata:")
        st.write(f"📍 **Location:** {st.session_state['metadata']['location']}")
        st.write(f"☁️ **Weather:** {st.session_state['metadata']['weather']}")
        st.write(f"⏰ **Time of Recording:** {st.session_state['metadata']['time']}")
        st.write(f"📝 **Notes:** {st.session_state['metadata']['notes']}")

    # Navigation Buttons
    if st.button("Back to Spectrogram 📊"):
        st.session_state["page"] = "📊 Spectrogram"
        st.rerun()

    if st.button("Back to Analysis 🔍"):
        st.session_state["page"] = "🔍 Analysis"
        st.rerun()