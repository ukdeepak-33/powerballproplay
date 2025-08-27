from flask import Flask, jsonify, render_template
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# --- File Paths ---
# Get the absolute path to the directory containing this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Set the path to the templates folder, one directory up
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')
# Define the path for the saved model file
MODEL_PATH = "model.pkl"

# --- Model Loading and Training Logic ---
def train_and_save_model():
    """
    Trains a simple RandomForestClassifier and saves it to a file.
    This function is called only if the model file does not exist.
    """
    print("Training a new model...")
    # Create a fake dataset for demonstration purposes
    # X represents 5 white balls (1-69), y is the Powerball (1-26)
    X = np.random.randint(1, 70, size=(500, 5))
    y = np.random.randint(1, 27, size=(500,))

    # Initialize and train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model to the file system
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")
    return model

# Check for the model file and load it, or train a new one
# This code runs as soon as the app.py file is imported by gunicorn
try:
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = joblib.load(MODEL_PATH)
    else:
        print("Model file not found. Starting training process...")
        model = train_and_save_model()
except Exception as e:
    print(f"Error during model loading/training: {e}")
    # Set model to None so the API endpoint can handle the error gracefully
    model = None

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# --- Routes ---
@app.route("/")
def index():
    """
    Serves the main page of the application.
    """
    return render_template("index.html")

@app.route("/api/generate_ml_numbers")
def generate_ml_numbers():
    """
    API endpoint to generate a Powerball number combination
    using the loaded machine learning model.
    """
    # Check if the model was loaded successfully before using it
    if model is None:
        return jsonify({"success": False, "message": "Models are not trained or loaded yet."}), 500

    try:
        # Generate 5 random white balls as input for the model
        X_new = np.random.randint(1, 70, size=(1, 5))

        # Predict the Powerball number using the model
        predicted_powerball = int(model.predict(X_new)[0])

        # Sort the white balls for a clean output
        white_balls = sorted(list(X_new[0]))

        return jsonify({
            "success": True,
            "white_balls": white_balls,
            "powerball": predicted_powerball
        })
    except Exception as e:
        # Handle any runtime errors during prediction
        print(f"Error during number generation: {e}")
        return jsonify({"success": False, "message": "An error occurred during number generation."}), 500
