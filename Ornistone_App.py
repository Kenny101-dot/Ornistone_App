import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display




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
    st.title("🔍 AI Analysis on its Endangered Status")

    if "spectrogram" in st.session_state:
        st.success("📊 Spectrogram loaded, ready to analyze!")

        MODEL_PATH = "resnet_bird_224x224_round14.pth"  # Ensure the model file is in the same directory

        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image
        import matplotlib.pyplot as plt
        import librosa.display

        # Function to load the model
        @st.cache_resource()  # Cache the model so it loads only once
        def load_model():  # PUT INTO DIFFERENT PYTHON SCRIPT
            model = models.resnet50(weights=None)  # Don't load default weights
    
            # Recreate the exact FC layers used in training
            model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 150),
                nn.ReLU(),
                nn.Linear(150, 10),
                nn.ReLU(),
                nn.Dropout(p=0.3), 
                nn.Linear(10, 3)  # Final 3-class classification layer
            )
    
            # Load trained weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
            model.eval()  # Set model to evaluation mode
            return model

        # Load the model
        model = load_model()
        st.success("✅ Model loaded successfully!")

        # Convert the spectrogram to an image
        def spectrogram_to_image(spectrogram):
            fig, ax = plt.subplots(figsize=(4, 4))
            librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", ax=ax)
            plt.axis('off')
            
            # Save the spectrogram as an image
            spectrogram_path = "spectrogram.png"
            plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            return spectrogram_path

        # Generate spectrogram image
        spectrogram_path = spectrogram_to_image(st.session_state["spectrogram"])
        st.image(spectrogram_path, caption="Generated Spectrogram", use_container_width=True)

        # Prepare image for ResNet50 model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image = Image.open(spectrogram_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform Prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Map prediction to Conservation Status
        class_labels = ["Least Concern", "Vulnerable", "Endangered"]
        prediction_label = class_labels[predicted_class]

        st.write(f"🎯 **Predicted Conservation Status: {prediction_label}**")
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
    notes = st.text_area("📝 Further Notes (e.g., bird behavior):")

############# Save Metadata to Database #############

# Function to save metadata to the database
def save_metadata(location, weather, time, notes):
    conn = sqlite3.connect("metadata.db")  
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metadata 
                 (location TEXT, weather TEXT, time TEXT, notes TEXT)''')
    c.execute("INSERT INTO metadata (location, weather, time, notes) VALUES (?, ?, ?, ?)", 
              (location, weather, time, notes))
    conn.commit()
    conn.close()

# Function to export metadata as CSV and trigger download
def export_metadata_to_csv():
    conn = sqlite3.connect("metadata.db")  
    df = pd.read_sql_query("SELECT * FROM metadata", conn)  
    conn.close()
    
    csv_path = "metadata_export.csv"
    df.to_csv(csv_path, index=False)  
    return csv_path

# Save metadata and trigger download
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