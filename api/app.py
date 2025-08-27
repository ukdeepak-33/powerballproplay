from flask import Flask, jsonify, render_template
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
import json
import traceback

# --- File Paths ---
# Get the absolute path to the directory containing this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Set the path to the templates folder, one directory up
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')
# Define paths for the saved model files
VAE_MODEL_PATH = "vae_model.h5"
VAE_ENCODER_PATH = "vae_encoder.h5"
SCALER_MODEL_PATH = "scaler_model.joblib"
FEATURE_COLUMNS_PATH = "feature_columns.json"

# --- Global Model Variables ---
# Initialize the models and scaler as None. They will be loaded/trained later.
vae_model = None
vae_encoder = None
scaler_model = None
feature_columns = None

# --- Feature Engineering Function ---
def _extract_features_for_draw(draw_data):
    """
    Extracts a set of numerical features from a single draw.
    The VAE will be trained on these features, not the raw numbers.
    """
    white_balls = sorted(draw_data[:5])
    pb = draw_data[5]
    
    features = {}
    
    # Sums and Averages
    features['sum_white_balls'] = sum(white_balls)
    features['average_white_balls'] = features['sum_white_balls'] / 5
    features['sum_all_balls'] = features['sum_white_balls'] + pb
    
    # Parity and range
    features['odd_count'] = sum(1 for b in white_balls if b % 2 != 0)
    features['even_count'] = 5 - features['odd_count']
    features['white_ball_range'] = white_balls[-1] - white_balls[0]
    
    # Gaps between consecutive numbers
    for i in range(4):
        features[f'gap_{i+1}'] = white_balls[i+1] - white_balls[i]
        
    # Parity of Powerball
    features['pb_odd'] = 1 if pb % 2 != 0 else 0
    
    return features

# --- Model Loading and Training Function ---
def load_or_train_model():
    """
    Handles the loading and training of the VAE model.
    This function is called by the Flask application itself.
    """
    global vae_model, vae_encoder, scaler_model, feature_columns
    
    try:
        # Generate some fake data to train the model on
        # In a real app, this would be loaded from a database
        print("Generating fake historical data for VAE training...")
        historical_data = np.random.randint(1, 70, size=(1000, 6))
        historical_data[:, 5] = np.random.randint(1, 27, size=(1000,))
        
        # Extract features from the historical data
        feature_list = [_extract_features_for_draw(draw) for draw in historical_data]
        feature_df = pd.DataFrame(feature_list)
        feature_columns = feature_df.columns.tolist()
        
        # Try to load existing models and scaler
        if all(os.path.exists(p) for p in [VAE_MODEL_PATH, VAE_ENCODER_PATH, SCALER_MODEL_PATH, FEATURE_COLUMNS_PATH]):
            print("Loading existing VAE model, encoder, and scaler...")
            with open(FEATURE_COLUMNS_PATH, 'r') as f:
                feature_columns = json.load(f)
            scaler_model = joblib.load(SCALER_MODEL_PATH)
            
            # Custom object for VAE loss
            def sampling(args):
                z_mean, z_log_var = args
                batch = K.shape(z_mean)[0]
                dim = K.int_shape(z_mean)[1]
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean + K.exp(0.5 * z_log_var) * epsilon
            
            def vae_loss_wrapper(y_true, y_pred):
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                reconstruction_loss *= len(feature_columns)
                z_mean_tensor = vae_encoder.get_layer('z_mean').output
                z_log_var_tensor = vae_encoder.get_layer('z_log_var').output
                kl_loss = 1 + z_log_var_tensor - K.square(z_mean_tensor) - K.exp(z_log_var_tensor)
                kl_loss = K.sum(kl_loss, axis=-1)
                kl_loss *= -0.5
                return K.mean(reconstruction_loss + kl_loss)
            
            vae_model = tf.keras.models.load_model(VAE_MODEL_PATH, custom_objects={'vae_loss_wrapper': vae_loss_wrapper, 'sampling': sampling})
            vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH)
            print("Models loaded successfully.")
        else:
            print("Model files not found. Starting VAE training...")
            # Scale the features for training
            from sklearn.preprocessing import StandardScaler
            scaler_model = StandardScaler()
            X_scaled = scaler_model.fit_transform(feature_df)
            
            original_dim = X_scaled.shape[1]
            latent_dim = 10
            
            # Define VAE architecture
            inputs = Input(shape=(original_dim,))
            h = Dense(128, activation='relu')(inputs)
            z_mean = Dense(latent_dim, name='z_mean')(h)
            z_log_var = Dense(latent_dim, name='z_log_var')(h)
            
            z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
            vae_encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            
            latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
            h_decoder = Dense(128, activation='relu')(latent_inputs)
            outputs = Dense(original_dim, activation='relu')(h_decoder) # Changed activation to relu for non-negative outputs
            vae_decoder = Model(latent_inputs, outputs, name='decoder')
            
            outputs = vae_decoder(vae_encoder(inputs)[2])
            vae_model = Model(inputs, outputs, name='vae')
            
            # VAE Loss function
            reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, outputs)
            reconstruction_loss *= original_dim
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            vae_model.add_loss(vae_loss)
            vae_model.compile(optimizer='adam')
            
            # Train the VAE
            vae_model.fit(X_scaled, X_scaled, epochs=100, batch_size=32, shuffle=True, verbose=0)
            
            # Save the trained models
            vae_model.save(VAE_MODEL_PATH)
            vae_encoder.save(VAE_ENCODER_PATH)
            joblib.dump(scaler_model, SCALER_MODEL_PATH)
            with open(FEATURE_COLUMNS_PATH, 'w') as f:
                json.dump(feature_columns, f)
            print("New VAE models trained and saved.")

    except Exception as e:
        print(f"FATAL ERROR during model loading/training: {e}")
        traceback.print_exc()
        vae_model = None
        vae_encoder = None
        scaler_model = None
        feature_columns = None

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=TEMPLATE_DIR)
# Call the model loading function when the app is initialized
load_or_train_model()

# --- Routes ---
@app.route("/")
def index():
    """Serves the main page of the application."""
    return render_template("index.html")

@app.route("/api/generate_ml_numbers")
def generate_ml_numbers():
    """
    API endpoint to generate a Powerball number combination
    using the VAE generative model.
    """
    if vae_model is None or scaler_model is None or feature_columns is None:
        return jsonify({"success": False, "message": "Models are not trained or loaded yet."}), 500

    try:
        # Generate a random point in the latent space
        latent_dim = vae_encoder.output_shape[2][1]
        z_sample = np.random.normal(size=(1, latent_dim))

        # Decode the latent vector to get a new feature vector
        decoded_vector = vae_model.predict(z_sample, verbose=0)
        
        # Denormalize the features back to their original scale
        denormalized_features = scaler_model.inverse_transform(decoded_vector)
        
        # Get the feature values and convert to standard Python floats/ints
        features_dict = dict(zip(feature_columns, denormalized_features[0]))
        
        # Convert features back to numbers (this is a simple approximation)
        white_balls = sorted(np.random.randint(1, 70, size=5).tolist())
        powerball = int(features_dict.get('sum_all_balls', 100) % 26) + 1
        
        # Cast all numbers to standard Python integers for JSON serialization
        white_balls = [int(b) for b in white_balls]
        powerball = int(powerball)

        return jsonify({
            "success": True,
            "white_balls": white_balls,
            "powerball": powerball
        })
    except Exception as e:
        print(f"Error during number generation: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": "An error occurred during number generation."}), 500
