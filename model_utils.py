# model_utils.py
import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import cv2
import sqlite3
import pandas as pd
import numpy as np



import torch

# Direkt das komplette Modell laden
model = torch.load("resnet_bird_224x224_roundWednesday.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()


# ⬇️ Hilfsfunktion zur Konvertierung eines Spektrogramms in ein Bild (224x224 RGB)
def spectrogram_to_image(S_dB):
    S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())  # Normalisieren auf [0,1]
    S_dB = (S_dB * 255).astype(np.uint8)  # In 8-Bit konvertieren
    S_dB_resized = cv2.resize(S_dB, (224, 224))
    S_dB_rgb = cv2.cvtColor(S_dB_resized, cv2.COLOR_GRAY2RGB)  # Graustufen → RGB

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(S_dB_rgb).unsqueeze(0)  # Batch-Dimension hinzufügen

# ⬇️ Vorhersage durchführen
def predict_spectrogram(S_dB):
    
    input_tensor = spectrogram_to_image(S_dB)  # Konvertiere das Spektrogramm
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]  # In Wahrscheinlichkeiten umwandeln
        predicted_class = torch.argmax(probabilities).item()

    # Klassenlabels für das Modell
    class_labels = ["Least Concern", "Vulnerable", "Endangered"]
    prediction_label = class_labels[predicted_class]

    # Dictionary mit Wahrscheinlichkeiten
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

