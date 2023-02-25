from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow       import keras
from tensorflow.keras import layers
import sys

def train_model():

    """
    Trains a convolutional neural network (CNN) to classify images using the training and validation data.
    
    Args:
        None, but the function expects the following command-line arguments respectivly:
            - train_path (str): path to the directory containing the training images, organized into subdirectories by class.
            - val_path (str): path to the directory containing the validation images, organized into subdirectories by class.
            - epochs_n (int): number of epochs to train the model for.
            - model_name (str): name of the file to save the trained model to.
            
    Returns:
        None, but the function saves the trained model as a file with the specified name.

    Example usage:
        train_model("data/train", "data/val", 10, "tf_model.h5")
    """


    train_path = sys.argv[1]
    val_path = sys.argv[2]
    epochs_n = sys.argv[3]
    model_name = sys.argv[4]

    # Define the input shape of the images
    input_shape = (224, 224, 3)

    # Define the model architecture
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    # Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    model.summary()


    # Create an instance of the ImageDataGenerator class with data augmentation parameters
    data_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load the training and validation data from a directory
    train_data = data_generator.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    val_data = data_generator.flow_from_directory(
        val_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    # Train the model
    model.fit(train_data, epochs=epochs_n, validation_data=val_data)

    # Save the model
    model.save(model_name)

train_model()