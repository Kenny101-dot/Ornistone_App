from matplotlib import transforms
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from PIL import Image
from model_utils import save_metadata, export_metadata_to_csv
import torch
import cv2
from model_utils import predict_spectrogram

# Streamlit App Setup
st.set_page_config(page_title="Bird Sound Classifier", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E2EAE2;
    }
    .stApp {
    secondaryBackgroundColor="#1daa0e"
    }
    .stButton>button {
        background-color: #1b6211;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #0C0B0B;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar für Navigation
with st.sidebar:
    st.title("Navigation")
    st.image("images/Ornithologist.png", use_container_width=True)
    pages = ["🏠 Welcome_Page", "📂 Audio-Upload", "📊 Spectrogram", "🔍 Analysis", "📝 Metadata Survey"]
    page = st.radio("Go to", pages, index=0)

#########################################################################################################################################
# Welcome_page
if page == "🏠 Welcome_Page":
   
    st.title("Hello fellow bird enthusiast! 🎶🐦")
    st.image("images/taube.png", width=800)
    st.write("#### **Recorded a sound? Click onto 📂 Audio-Upload on the left side! 🎵**")


    
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
        st.write("")
        st.write("#### **Click on 📊 Spectrogram on the left side!**")
       

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

        st.write(f"Looks like a valid audio file! 🎉 Maximum amplitude: {np.max(y)}.")
        # Mel-Spektrogramm berechnen
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S + 1e-6, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        st.pyplot(fig)

        st.session_state["spectrogram"] = S_dB
        st.write("#### **Click on 🔍 Analysis on the left side if you would like an estimation of the endangered status**")
    else:
        st.warning("Please upload a file first!")

#########################################################################################################################################
# Ornistone_App.py

elif page == "🔍 Analysis":
    st.title("🔍 AI Analysis on its Endangered Status")
    st.write("")
    if "spectrogram" in st.session_state:
        # 🔥 Direkt die Vorhersage aufrufen
        prediction_label, top3_probs = predict_spectrogram(st.session_state["spectrogram"])

        st.write(f"### 🎯 **Predicted Conservation Status: {prediction_label}**")
        st.write("")
        st.write("🔢 **Class Probabilities:**")
        for label, prob in top3_probs.items():
            st.write(f"- {label}: {prob * 100:.2f}%")

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("#### **Now insert further information about the recording at the 📝 Metadata Survey page**")
    else:
        st.warning("⚠️ No spectrogram available. Please upload an audio file first!")




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
    notes = st.text_area("📝 Further Notes:")

    if st.button("Save Metadata & Download CSV"):
            save_metadata(location, weather, str(time), notes)
            csv_file = export_metadata_to_csv()
            
            with open(csv_file, "rb") as file:
                st.download_button(
                    label="📥 Download CSV",
                    data=file,
                    file_name="metadata.csv",
                    mime="text/csv"
                )
