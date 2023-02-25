import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys


def train_model():

        """
        Train a ResNet50 model with pre-trained weights on a given dataset, and save the trained model to a file.

        Arguments:
                None, but the function expects the following command-line arguments respectivly:
                        - train_path (str): path to the directory containing the training images, organized into subdirectories by class.
                        - val_path (str): path to the directory containing the validation images, organized into subdirectories by class.
                        - epochs_n (int): number of epochs to train the model for.
                        - model_name (str): name of the file to save the trained model to.

        Returns:
                - None, but the functions saves the trained model as a file with specified name

        Example usage:
                train_model("data/train", "data/val", 10, "resnet.h5")
        """


        # Parse the command-line arguments
        train_path = sys.argv[1]
        val_path = sys.argv[2]
        epochs_n = sys.argv[3]
        model_name = sys.argv[4]

        # Load the pre-trained ResNet50 model with ImageNet weights
        base_model = ResNet50(weights='imagenet', include_top=False)

        # freeze all layers in the base model
        for layer in base_model.layers:
                layer.trainable = False

        # add classification layers on top of the base model
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        # Create new model by combining the base model and the classification layers
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model with Adam optimizer and categorical cross-entropy loss
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

        # Define data generators for train and validation sets with normalization
        data_gen = ImageDataGenerator(rescale=1./255)

        train_generator = data_gen.flow_from_directory(
                train_path,
                batch_size=8,
                class_mode='categorical')

        val_generator = data_gen.flow_from_directory(
                val_path,
                batch_size=8,
                class_mode='categorical')
                

        # train the model
        history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples//train_generator.batch_size,
                epochs=epochs_n,
                validation_data=val_generator,
                validation_steps=val_generator.samples//val_generator.batch_size,
                verbose = 1)

        # save the trained model
        model.save(model_name)

train_model()

