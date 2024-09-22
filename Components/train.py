import os
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from model_builder import build_cnn_model  # Assuming build_cnn_model is correctly implemented in model_builder


def preprocess_data(df):
    print("Preprocessing data...")

    # Drop unnecessary columns, convert types
    df = df.drop(columns=['Open Time', 'Close Time', 'Ignore'])  # If those columns are present
    df = df.astype('float32')  # Convert all columns to float32

    return df


def main():
    # Load config
    config_path = os.path.join('C:\\Users\\Arun2\\Documents\\Project\\Trading Strat\\Config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load data
    print("Loading data...")
    data_path = config['data']['path']
    df = pd.read_csv(data_path)
    print(f"Data loaded with {len(df)} rows")

    # Preprocess data
    print("Starting data preprocessing...")
    df = preprocess_data(df)
    print("Data preprocessing completed")

    # Separate features and labels
    X = df[config['data']['features']].values
    y = df[config['data']['label']].values

    # Define the TimeseriesGenerator for training and validation data
    sequence_length = config['data']['sequence_length']
    batch_size = config['model']['batch_size']
    print(f"Generating sequences with sequence length: {sequence_length}")

    generator = TimeseriesGenerator(X, y, length=sequence_length, batch_size=batch_size)

    # Split into training and validation sets
    num_validation_samples = int(len(generator) * config['training']['validation_split'])
    training_generator = TimeseriesGenerator(X, y, length=sequence_length, batch_size=batch_size,
                                             end_index=len(X) - num_validation_samples - 1)
    validation_generator = TimeseriesGenerator(X, y, length=sequence_length, batch_size=batch_size,
                                               start_index=len(X) - num_validation_samples)

    # Build the CNN model
    input_shape = (sequence_length, len(config['data']['features']), 1)
    print(f"Building model with input shape: {input_shape}")
    model = build_cnn_model(
        input_shape=input_shape,
        num_filters=config['model']['num_filters'],
        kernel_size=tuple(config['model']['kernel_size']),
        pool_size=tuple(config['model']['pool_size']),
        dropout_rate=config['model']['dropout_rate'],
        activation=config['model']['activation'],
        output_activation=config['model']['output_activation']
    )

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['optimizer']['learning_rate']),
        loss=config['model']['loss_function'],
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Add callbacks
    callbacks = []
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping']['patience'],
                                       verbose=1)
        callbacks.append(early_stopping)

    if config['training']['save_best_model']:
        checkpoint_path = config['training']['model_checkpoint_path']
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
        callbacks.append(model_checkpoint)

    # Train the model with tqdm for progress tracking
    print("Training the model...")
    epochs = config['model']['epoch']
    for epoch in tqdm(range(epochs), desc="Training progress"):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=1,  # Run one epoch at a time to track progress
            callbacks=callbacks,
            verbose=1
        )
        print(f"Completed epoch {epoch + 1}")


if __name__ == "__main__":
    main()
