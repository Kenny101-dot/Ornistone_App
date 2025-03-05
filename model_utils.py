import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
import sqlite3
import pandas as pd

MODEL_PATH = "resnet_bird_224x224_round14.pth"

# Load ResNet50 model with custom classifier
def load_model():
    model = models.resnet50(weights=None)  # Don't load default weights

   # Recreate the exact FC layers used in training
    model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 150),
                nn.ReLU(),
                nn.Linear(150, 10),
                nn.ReLU(),
                nn.Dropout(0.3), 
                nn.Linear(10, 3)  # Final 3-class classification layer
    )

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
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

# Perform prediction
def predict_spectrogram(model, spectrogram_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(spectrogram_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    class_labels = ["Least Concern", "Vulnerable", "Endangered"]
    return class_labels[predicted_class]

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
