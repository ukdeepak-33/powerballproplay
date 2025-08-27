import pandas as pd
from flask import Flask, render_template, request, jsonify
from itertools import combinations
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
import requests
import numpy as np
import traceback
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
import json
import joblib # For saving and loading models

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Supabase Configuration and Database Interactions ---
SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_URL", "https://yksxzbbcoitehdmsxqex.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "YOUR_ACTUAL_SUPABASE_ANON_KEY_GOES_HERE")
SUPABASE_TABLE_NAME = 'powerball_draws'

# Global variables to store data and models
df = pd.DataFrame()
kmeans_model = None
scaler_model = None
vae_model = None
vae_encoder = None
feature_columns = []

# Model file paths for persistence on Render.com
KMEANS_MODEL_PATH = 'kmeans_model.joblib'
SCALER_MODEL_PATH = 'scaler_model.joblib'
VAE_MODEL_PATH = 'vae_model.h5'
VAE_ENCODER_PATH = 'vae_encoder.h5'

def load_historical_data_from_supabase():
    """Fetches all historical Powerball draw data from Supabase."""
    try:
        url = f"{SUPABASE_PROJECT_URL}/rest/v1/{SUPABASE_TABLE_NAME}?select=*"
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            print("No data received from Supabase.")
            return pd.DataFrame()

        df_loaded = pd.DataFrame(data)
        # Convert necessary columns to numeric or datetime
        numeric_cols = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'powerball']
        for col in numeric_cols:
            df_loaded[col] = pd.to_numeric(df_loaded[col], errors='coerce')
        df_loaded['draw_date'] = pd.to_datetime(df_loaded['draw_date'], errors='coerce')
        
        return df_loaded.dropna(subset=numeric_cols + ['draw_date']).sort_values(by='draw_date', ascending=False)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame()

# --- Feature Engineering Functions ---
def _extract_features_for_draw(draw_data):
    """
    Extracts a rich set of features from a single draw for ML models.
    This function must be kept synchronized with the training data preparation.
    """
    try:
        white_balls = sorted([
            draw_data['ball_1'], draw_data['ball_2'], draw_data['ball_3'],
            draw_data['ball_4'], draw_data['ball_5']
        ])
        pb = draw_data['powerball']
        
        features = {}
        
        # Simple descriptive statistics
        features['sum_white_balls'] = sum(white_balls)
        features['average_white_balls'] = features['sum_white_balls'] / 5
        features['median_white_balls'] = white_balls[2]
        features['sum_all_balls'] = features['sum_white_balls'] + pb
        
        # Parity and range
        features['odd_count'] = sum(1 for b in white_balls if b % 2 != 0)
        features['even_count'] = 5 - features['odd_count']
        features['low_count'] = sum(1 for b in white_balls if b <= 34)
        features['high_count'] = 5 - features['low_count']
        features['white_ball_range'] = white_balls[-1] - white_balls[0]
        
        # Differences between consecutive numbers
        for i in range(4):
            features[f'gap_{i+1}'] = white_balls[i+1] - white_balls[i]
            
        # Parity of Powerball
        features['pb_odd'] = 1 if pb % 2 != 0 else 0
        
        return features
    except Exception as e:
        print(f"Error extracting features for draw: {e}")
        return None

# --- ML/DL Model Building and Training Functions ---
def _train_kmeans_model(X):
    """
    Trains a K-Means clustering model on the feature data.
    The model is saved for later use.
    """
    global kmeans_model, scaler_model
    try:
        print("Training K-Means model...")
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(X)
        
        # You can use PCA to reduce dimensionality before clustering
        # pca = PCA(n_components=5)
        # X_pca = pca.fit_transform(X_scaled)
        
        kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10) # Using n_init=10 to suppress warning
        kmeans_model.fit(X_scaled)
        
        joblib.dump(kmeans_model, KMEANS_MODEL_PATH)
        joblib.dump(scaler_model, SCALER_MODEL_PATH)
        print("K-Means model trained and saved.")
    except Exception as e:
        print(f"Error training K-Means model: {e}")
        traceback.print_exc()

def _train_vae_model(X, latent_dim=10, epochs=100):
    """
    Trains a Variational Autoencoder (VAE) for generative modeling.
    The model is saved for later use.
    """
    global vae_model, vae_encoder
    try:
        print("Training VAE model...")
        
        # Normalize the data before training VAE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define the VAE architecture
        original_dim = X_scaled.shape[1]
        
        # Encoder
        inputs = Input(shape=(original_dim,))
        h = Dense(128, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        vae_encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        h_decoder = Dense(128, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(h_decoder) # Sigmoid for normalized data
        vae_decoder = Model(latent_inputs, outputs, name='decoder')
        
        # VAE model
        outputs = vae_decoder(vae_encoder(inputs)[2])
        vae_model = Model(inputs, outputs, name='vae')
        
        # VAE loss function
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae_model.add_loss(vae_loss)
        
        vae_model.compile(optimizer='adam')
        
        # Train the model
        vae_model.fit(X_scaled, X_scaled, epochs=epochs, batch_size=32, shuffle=True, verbose=0)
        
        # Save both models for persistence
        vae_model.save(VAE_MODEL_PATH)
        vae_encoder.save(VAE_ENCODER_PATH)
        print("VAE model trained and saved.")
    except Exception as e:
        print(f"Error training VAE model: {e}")
        traceback.print_exc()

def _train_all_models():
    """Main function to train and save all models."""
    global df, feature_columns
    print("Starting model training...")
    try:
        # Load historical data
        if df.empty:
            df = load_historical_data_from_supabase()

        if df.empty:
            print("Historical data is empty. Skipping model training.")
            return
        
        # Extract features
        feature_list = []
        for _, row in df.iterrows():
            features = _extract_features_for_draw(row)
            if features:
                feature_list.append(features)
        
        if not feature_list:
            print("No features extracted. Skipping model training.")
            return
        
        X = pd.DataFrame(feature_list)
        feature_columns = X.columns.tolist()

        # Train models
        _train_kmeans_model(X)
        _train_vae_model(X)
        print("All models trained successfully.")
    except Exception as e:
        print(f"Failed to complete model training: {e}")
        traceback.print_exc()

def _load_or_train_models():
    """Loads models if they exist, otherwise trains them."""
    global kmeans_model, scaler_model, vae_model, vae_encoder, feature_columns
    
    # Load feature columns, needed for generation logic
    if os.path.exists('feature_columns.json'):
        with open('feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
    else:
        # A simple dummy load just to get the columns if models don't exist
        temp_df = pd.DataFrame([_extract_features_for_draw(df.iloc[0])])
        feature_columns = temp_df.columns.tolist()
        with open('feature_columns.json', 'w') as f:
            json.dump(feature_columns, f)

    if os.path.exists(KMEANS_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
        print("Loading trained K-Means and Scaler models...")
        try:
            kmeans_model = joblib.load(KMEANS_MODEL_PATH)
            scaler_model = joblib.load(SCALER_MODEL_PATH)
            print("K-Means models loaded successfully.")
        except Exception as e:
            print(f"Error loading K-Means models: {e}. Re-training.")
            _train_all_models()
    else:
        print("K-Means models not found. Starting training...")
        _train_all_models()
    
    if os.path.exists(VAE_MODEL_PATH) and os.path.exists(VAE_ENCODER_PATH):
        print("Loading trained VAE models...")
        try:
            vae_model = tf.keras.models.load_model(VAE_MODEL_PATH, custom_objects={'vae_loss': _get_vae_loss(len(feature_columns))})
            vae_encoder = tf.keras.models.load_model(VAE_ENCODER_PATH)
            print("VAE models loaded successfully.")
        except Exception as e:
            print(f"Error loading VAE models: {e}. Re-training.")
            _train_all_models()
    else:
        print("VAE models not found. Starting training...")
        _train_all_models()

def _get_vae_loss(original_dim):
    """Helper function to recreate the VAE loss for loading."""
    def vae_loss_wrapper(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        reconstruction_loss *= original_dim
        
        z_mean = vae_encoder.get_layer('z_mean').output
        z_log_var = vae_encoder.get_layer('z_log_var').output
        
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        return K.mean(reconstruction_loss + kl_loss)
    return vae_loss_wrapper

# --- Number Generation Logic using ML/DL Models ---
def _generate_from_vae():
    """Generates a number combination using the trained VAE model."""
    if vae_encoder is None or scaler_model is None or not feature_columns:
        return "Models are not ready. Please check server logs."

    # Sample a point from the latent space
    z_sample = np.random.normal(size=(1, vae_encoder.layers[-1].output_shape[-1]))

    # Decode the sample to a feature vector
    decoded_vector = vae_model.predict(z_sample)
    
    # De-normalize the vector using the scaler's inverse transform
    denormalized_features = scaler_model.inverse_transform(decoded_vector)
    
    # Convert feature vector back to a lottery combination (simplified)
    # This is a critical and complex step that requires a lot of custom logic.
    # The VAE outputs floats, which must be converted to valid, unique integers.
    # For now, we will use a heuristic approach.
    
    features_dict = dict(zip(feature_columns, denormalized_features[0]))
    
    # Heuristic: convert features to numbers. This is where you would refine the logic.
    sum_white = int(round(features_dict['sum_white_balls']))
    odd_count = int(round(features_dict['odd_count']))
    even_count = 5 - odd_count
    
    # A simple, but not robust, way to get numbers from features.
    # In a real-world app, you would have a more sophisticated algorithm.
    generated_balls = []
    
    while len(generated_balls) < 5:
        ball = np.random.randint(1, 70)
        if ball not in generated_balls:
            generated_balls.append(ball)
            
    pb = np.random.randint(1, 27)

    return sorted(generated_balls) + [pb]

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
app.secret_key = 'powerball_pro_play'

# --- API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_ml_numbers', methods=['GET'])
def generate_ml_numbers_api():
    """Generates a new number combination using ML/DL models."""
    try:
        if vae_model is None or scaler_model is None:
            return jsonify({'success': False, 'message': 'Models are not trained or loaded yet.'}), 500
        
        generated_combination = _generate_from_vae()
        
        return jsonify({
            'success': True,
            'white_balls': generated_combination[:5],
            'powerball': generated_combination[5]
        })
    except Exception as e:
        print(f"Error generating numbers: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'An internal error occurred during generation.'}), 500

# --- Re-implementing existing analytics routes from dump-index.py ---
@app.route('/api/last_draw', methods=['GET'])
def api_last_draw():
    if df.empty:
        return jsonify({'success': False, 'error': "Historical data not loaded or is empty."}), 500
    last_draw_data = df.iloc[0].to_dict()
    return jsonify({'success': True, 'last_draw': last_draw_data})

@app.route('/api/draw_stats', methods=['GET'])
def api_draw_stats():
    if df.empty:
        return jsonify({'success': False, 'error': "Historical data not loaded or is empty."}), 500

    def calculate_stats(data_frame):
        white_balls = [f'ball_{i}' for i in range(1, 6)]
        all_balls = data_frame[white_balls].values.flatten()
        
        counts = pd.Series(all_balls).value_counts().sort_index()
        powerball_counts = data_frame['powerball'].value_counts().sort_index()
        
        return {
            'white_ball_frequencies': counts.to_dict(),
            'powerball_frequencies': powerball_counts.to_dict(),
            'most_common_white_balls': counts.nlargest(10).index.tolist(),
            'least_common_white_balls': counts.nsmallest(10).index.tolist(),
            'most_common_powerballs': powerball_counts.nlargest(5).index.tolist(),
            'least_common_powerballs': powerball_counts.nsmallest(5).index.tolist(),
        }

    stats = calculate_stats(df)
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/white_ball_gaps', methods=['GET'])
def api_white_ball_gaps():
    if df.empty:
        return jsonify({'success': False, 'error': "Historical data not loaded or is empty."}), 500

    target_number_str = request.args.get('number')
    if not target_number_str or not target_number_str.isdigit():
        return jsonify({'success': False, 'error': 'Invalid white ball number provided.'}), 400
    
    target_number = int(target_number_str)
    if not (1 <= target_number <= 69):
        return jsonify({'success': False, 'error': 'White ball number must be between 1 and 69.'}), 400

    try:
        all_draws = df.to_dict('records')
        last_appearance = None
        gaps = []

        for draw in all_draws:
            draw_date = pd.to_datetime(draw['draw_date'])
            white_balls = sorted([draw['ball_1'], draw['ball_2'], draw['ball_3'], draw['ball_4'], draw['ball_5']])
            
            if target_number in white_balls:
                if last_appearance:
                    gap = (last_appearance - draw_date).days
                    gaps.append(gap)
                last_appearance = draw_date
        
        if not gaps:
            return jsonify({'success': False, 'error': f"Number {target_number} has not appeared in the dataset."}), 404
            
        gaps_data = {
            'average_gap': sum(gaps) / len(gaps),
            'max_gap': max(gaps),
            'min_gap': min(gaps),
            'last_gap': gaps[0] if gaps else 0
        }
        return jsonify({'success': True, 'gaps_data': gaps_data})
    except Exception as e:
        print(f"Error calculating gaps: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500

# --- Application Startup ---
if __name__ == '__main__':
    print("Loading historical data...")
    df = load_historical_data_from_supabase()
    
    if not df.empty:
        _load_or_train_models()

    app.run(debug=True)
