import os
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model


def preprocess_data(df):
    # Drop unnecessary columns and convert types if needed
    df = df.drop(columns=['Open Time', 'Close Time', 'Ignore'], errors='ignore')
    df = df.astype('float32')
    return df


def main():
    # Load config
    config_path = os.path.join('C:\\Users\\Arun2\\Documents\\Project\\Trading Strat\\Config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load new data for predictions
    data_path = config['data']['path']
    df = pd.read_csv(data_path)
    print(f"Data loaded with {len(df)} rows for prediction")

    # Preprocess the data
    df = preprocess_data(df)

    # Separate features for prediction
    X = df[config['data']['features']].values

    # Create sequences for prediction
    sequence_length = config['data']['sequence_length']
    num_samples = len(X) - sequence_length + 1  # Adjust to create valid sequences

    # Create a 3D array for the model input
    X_sequences = np.array([X[i:i + sequence_length] for i in range(num_samples)])

    # Reshape for model input
    X_reshaped = X_sequences.reshape((-1, sequence_length, len(config['data']['features']), 1))

    # Load the trained model
    model_checkpoint_path = config['training']['model_checkpoint_path']
    model = load_model(model_checkpoint_path)

    # Make predictions
    predictions = model.predict(X_reshaped)
    predictions = np.squeeze(predictions)  # Remove single-dimensional entries

    # Output predictions
    output_path = os.path.join('C:\\Users\\Arun2\\Documents\\Project\\Trading Strat\\Components', 'predictions.csv')
    pd.DataFrame(predictions, columns=['Predicted_Label']).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
