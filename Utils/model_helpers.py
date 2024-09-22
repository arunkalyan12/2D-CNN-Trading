import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

def build_model(input_shape, num_filters, kernel_size, pool_size, dropout_rate, activation, output_activation):
    """
    Build a CNN model with the given parameters.

    Parameters:
    - input_shape (tuple): Shape of the input data (sequence_length, num_features, 1).
    - num_filters (list of int): Number of filters for each Conv2D layer.
    - kernel_size (int or tuple): Size of the convolutional kernels.
    - pool_size (int or tuple): Size of the max pooling windows.
    - dropout_rate (float): Dropout rate for the Dropout layer.
    - activation (str): Activation function to use in Conv2D and Dense layers.
    - output_activation (str): Activation function for the output layer.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # Should match (sequence_length, num_features, 1)

    # Add convolutional layers
    model.add(Conv2D(filters=num_filters[0], kernel_size=kernel_size, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(filters=num_filters[1], kernel_size=kernel_size, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(filters=num_filters[2], kernel_size=kernel_size, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation=output_activation))

    return model

def compile_model(model, learning_rate, beta_1, beta_2, epsilon):
    """
    Compile the model with the specified optimizer parameters.

    Parameters:
    - model (tf.keras.Model): Keras model to compile.
    - learning_rate (float): Learning rate for the Adam optimizer.
    - beta_1 (float): Beta_1 parameter for the Adam optimizer.
    - beta_2 (float): Beta_2 parameter for the Adam optimizer.
    - epsilon (float): Epsilon parameter for the Adam optimizer.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_pretrained_model(model_path, learning_rate, beta_1, beta_2, epsilon):
    """
    Load a saved model and recompile it with a fresh optimizer.

    Parameters:
    - model_path (str): Path to the saved model.
    - learning_rate (float): Learning rate for the Adam optimizer.
    - beta_1 (float): Beta_1 parameter for the Adam optimizer.
    - beta_2 (float): Beta_2 parameter for the Adam optimizer.
    - epsilon (float): Epsilon parameter for the Adam optimizer.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    # Load the model without compiling
    model = tf.keras.models.load_model(model_path, compile=False)

    # Recompile the model with a fresh Adam optimizer using the provided parameters
    return compile_model(model, learning_rate, beta_1, beta_2, epsilon)
