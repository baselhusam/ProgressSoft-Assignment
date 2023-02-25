from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def show_pred():

    """
    Load a saved model and use it to predict whether a person is wearing a mask or not in a given image. 
    The function displays the image with the predicted class label ('With Mask' or 'Without Mask') 
    using Matplotlib.

    Parameters:
        - None, but expect the following from the command-line arguments:
            - sys.argv[1]: the path to the image to be predicted
            - sys.argv[2]: the path to the saved model

    """

    # load the values for the parameters
    img_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load the model
    loaded_model = load_model(model_path)

    # Load the image and resize it to the input size of the model
    img = image.load_img(img_path, target_size=(224,224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Make a prediction with the model
    prediction = loaded_model.predict(img_arr)

    # Get the predicted class label
    class_label = np.argmax(prediction)

    # make the predictions as string label
    pred = "With Mask" if class_label == 0 else "Without Mask"

    # Display the image with the predicted class label
    plt.imshow(img)
    plt.axis('off')
    plt.title('Prediction: {}'.format(pred))
    plt.show()

show_pred()