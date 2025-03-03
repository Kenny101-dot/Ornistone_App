import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf



st.title("Bird Sound Classifier 🎶🐦")
st.write("Upload an audio file to classify the bird species.")

# DATA UPLOAD
uploaded_file = st.file_uploader("Upload a sound file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # SSAVE AS .ogg
    file_path = f"temp_audio.ogg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path, format="audio/ogg")  # PLAY THE AUDIO FILE
    
    st.write("✅ File saved successfully!")


# LOAD DATA WITH LIBROSA
y, sr = librosa.load(file_path, sr=22050)  # Sample-Rate setzen

# SHOW WAVEFORM
st.write("📊 Waveform of the audio signal:")
fig, ax = plt.subplots(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr, ax=ax)
st.pyplot(fig)

# CREATING SPECTROGRAMS
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

# VISUALIZING SPECTROGRAM
st.write("📊 Mel Spectrogram:")
fig, ax = plt.subplots(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
st.pyplot(fig)