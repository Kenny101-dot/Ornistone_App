import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

st.title("Bird Sound Classifier 🎶🐦")

uploaded_file = st.file_uploader("Upload a sound file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    file_path = "temp_audio.ogg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/ogg")
    st.write("✅ File saved successfully!")

    # load with librosa
    y, sr = librosa.load(file_path, sr=22050)
    
    # error diagnosis
    if len(y) == 0:
        st.error("Error: Empty audio file")
        st.stop()
    st.write(f"Maximum amplitude: {np.max(y)}")

    # show spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S + 1e-6, ref=np.max)  # Skalenproblem vermeiden

    st.write("📊 Mel Spectrogram:")
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    st.pyplot(fig)


# NOW FOR THE CLASSIFICATION PART, LOADING THE MODEL
MODEL_PATH = "bird_model.pth"  # INSERT RIGHT PATH
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()  # model in inference mode

# SPectrogram into right format
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S = librosa.power_to_db(S, ref=np.max)

# Batch Dimension
S = np.expand_dims(S, axis=0)  # iff CNN expected (1, height, width)

# PREDICTION
S_tensor = torch.tensor(S).unsqueeze(0)  
prediction = model(S_tensor)





