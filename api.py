from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

# Initialiser FastAPI
app = FastAPI()

# Charger le modèle
model_path = "/Users/mansour/Desktop/computer_vusion_project/my_model_fruit360.keras"
model = load_model(model_path)

# Récupérer les classes à partir du modèle (si possible)
class_indices = model.class_names if hasattr(model, "class_names") else None

# Fonction de prétraitement d'image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension batch
    img_array = img_array / 255.0  # Normalisation
    return img_array

# Endpoint de test
@app.get("/")
def home():
    return {"message": "API de classification d'images avec FastAPI"}

# Endpoint de prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    
    # Sauvegarder temporairement l'image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prétraiter et faire la prédiction
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Supprimer le fichier temporaire
    os.remove(file_path)

    return {"predicted_class": int(predicted_class)}
