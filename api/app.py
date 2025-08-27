from flask import Flask, jsonify, render_template
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import traceback

# --- File Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')
MODEL_PATH = "model.pkl"

# --- Global Model Variable ---
# Initialize the model as None. It will be loaded/trained later.
model = None

# --- Model Loading and Training Function ---
def load_or_train_model():
    """
    Handles the loading and training of the ML model.
    This function is called by the Flask application itself.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Starting training process...")
            # Create a fake dataset for demonstration
            X = np.random.randint(1, 70, size=(500, 5))
            y = np.random.randint(1, 27, size=(500,))
            
            # Initialize and train the classifier
            trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
            trained_model.fit(X, y)
            
            # Save the trained model to the file system
            joblib.dump(trained_model, MODEL_PATH)
            model = trained_model
            print("Model trained and saved.")
    except Exception as e:
        print(f"FATAL ERROR during model loading/training: {e}")
        traceback.print_exc()
        model = None

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Call the model loading function as soon as the app is created
# This ensures the model is ready before any requests are handled
load_or_train_model()

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
        # Cast the result to a standard Python int
        predicted_powerball = int(model.predict(X_new)[0])

        # Sort the white balls and cast them to standard Python integers
        white_balls = sorted(list(X_new[0].astype(int)))

        return jsonify({
            "success": True,
            "white_balls": white_balls,
            "powerball": predicted_powerball
        })
    except Exception as e:
        # Catch any runtime errors and provide a clear message
        print(f"Error during number generation: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": "An error occurred during number generation."}), 500
