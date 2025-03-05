# Description: Utility functions for loading the model, converting spectrogram to image, and making predictions.
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display


# Load the entire model directly
def load_model():
    model = torch.load("resnet_bird_224x224_round14.pth", map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    return model

# Convert spectrogram to image
def spectrogram_to_image(spectrogram):
    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis="time", y_axis="mel", ax=ax)
    plt.axis('off')

    # Save as image
    spectrogram_path = "spectrogram.png"
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return spectrogram_path

# Perform prediction with probability output
def predict_spectrogram(spectrogram_path):
    model = load_model()  # Ensure model is loaded once

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(spectrogram_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)  # Raw logits
        probabilities = torch.softmax(output, dim=1)[0]  # Convert to probabilities
        predicted_class = torch.argmax(probabilities).item()

    # Map prediction to Conservation Status
    class_labels = ["Least Concern", "Vulnerable", "Endangered"]
    prediction_label = class_labels[predicted_class]

    # Dictionary with class probabilities
    top3_probs = {class_labels[i]: probabilities[i].item() for i in range(3)}

    return prediction_label, top3_probs


########################### saving metadata functions ########################################

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
