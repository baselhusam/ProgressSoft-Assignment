import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class FaceMaskClassifier:

    """
    A class to predict whether a person is wearing a face mask or not.

    Attributes:
        img_path (str): The path to the image to be predicted.
        model_path (str): The path to the trained model to be used for prediction.

    Methods:
        predict(): Load the trained model and predict whether the person in the image is wearing a mask or not.

    """


    def __init__(self, img_path, model_path):
        
        """
        Constructs all the necessary attributes for the FaceMaskClassifier object.

        Parameters:
           img_path (str): The path to the image to be predicted.
           model_path (str): The path to the trained model to be used for prediction.

        """

        self.img_path = img_path
        self.model_path = model_path


    def predict(self):

        """
        Load the trained model and predict whether the person in the image is wearing a mask or not.

        Returns:
            label_pred (bool): True if the person is wearing a mask, False otherwise.

        """

        # Load the model
        model = load_model(self.model_path)

        # Load the image and resize it to the input size of the model
        img = image.load_img(self.img_path, target_size=(224,224))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # Make Prediction
        prediction = model.predict(img_arr)

        # Get the predicted class
        class_label = np.argmax(prediction)

        # Get the label of the predicted class
        label_pred = True if class_label == 0 else False

        return label_pred
