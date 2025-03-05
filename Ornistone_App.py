import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from PIL import Image
from model_utils import load_model, spectrogram_to_image, predict_spectrogram
from model_utils import save_metadata, export_metadata_to_csv
import torch

# Streamlit App Setup
st.set_page_config(page_title="Bird Sound Classifier", layout="wide")

# Sidebar für Navigation
with st.sidebar:
    st.title("Navigation")
    st.image("images/Ornithologist.png", use_container_width=True)
    pages = ["🏠 Welcome_Page", "📂 Audio-Upload", "📊 Spectrogram", "🔍 Analysis", "📝 Metadata Survey"]
    page = st.radio("Go to", pages, index=0)

#########################################################################################################################################
# Welcome_page
if page == "🏠 Welcome_Page":
    st.title("🎶🐦 What is this bird?")
    st.write("Hello fellow Bird Enthusiast! Pleased to have you. Did you ever wander through the forest and heard a perculiar song, one which you cannot categorize? Well, this App helps identifying it! With the help of a preptrained AI model, we can predict the species of the bird. In the current version it is solely able to tell you if it is endangered or not. Further updates are to come 🦜🌳")
    st.write("")  
 #button
    if st.button("Recorded a sound? Click onto Upload audio file on the left side! 🎵"):  # Fixed emoji
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
        
        if st.button("Click on 📊 Spectrogram on the left side!"):  # Fixed button text
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

        st.write(f"Maximum amplitude: {np.max(y)}. This is a validation that there is sound in the file.")
        st.write(f"Looks like a valid audio file! 🎉")

        # Mel-Spektrogramm berechnen
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S + 1e-6, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        st.pyplot(fig)

        st.session_state["spectrogram"] = S_dB

        if st.button("Click on 🔍 Analysis on the left side if you would like an estimation of the endangered status"):  # Fixed button text
            st.session_state["page"] = "🔍 Analysis"
            st.rerun()
    else:
        st.warning("Please upload a file first!")

#########################################################################################################################################
#  Analysis Page
elif page == "🔍 Analysis":
    st.title("🔍 AI Analysis on its Endangered Status")

    if "spectrogram" in st.session_state:
        st.success("📊 Spectrogram loaded, ready to analyze!")

        # Convert spectrogram to image
        spectrogram_path = spectrogram_to_image(st.session_state["spectrogram"])
        #st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)

        # Perform prediction
        prediction_label, top3_probs = predict_spectrogram(spectrogram_path)

        # Display results
        st.write(f"🎯 **Predicted Conservation Status: {prediction_label}**")
        st.write("🔢 **Class Probabilities:**")
        for label, prob in top3_probs.items():
            st.write(f"- {label}: {prob * 100:.2f}%")
    else:
        st.warning("⚠ No Spectrogram found! Please upload a file first.")





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
