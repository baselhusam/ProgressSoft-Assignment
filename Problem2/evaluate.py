from tensorflow.keras.preprocessing.image   import load_img, img_to_array
from tensorflow.keras.models                import load_model

import os
import numpy as np
import sys

def evaluate():

    """
    Evaluate the model on the test set and calculate accuracy, precision, recall, and f1-score.

    Args:
        None, but expect the following from the command-line:
            - test_path: the test data path to make predictions on it
            - model_path: the model path to evaluate it

    Returns:
        precision (float): precision score for the model
        recall (float): recall score for the model
        f1_score (float): f1-score for the model
        accuracy (float): the accuracy for the model

    """

    # load the values for the parameters
    test_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load the saved model
    loaded_model = load_model(model_path)

    # Define the input shape of the images
    input_shape = (224, 224, 3)

    # Load the test data from a directory
    test_with_mask_images = os.listdir(os.path.join(test_path, "WithMask"))
    test_without_mask_images = os.listdir(os.path.join(test_path, "WithoutMask"))

    # predictions list
    with_mask_pred, without_mask_pred = [], []

    # Loop through the with mask images and make predictions
    for image_name in test_with_mask_images:

        # Load the image and convert it to a NumPy array
        image = load_img(os.path.join(test_path, "WithMask", image_name), target_size=input_shape[:2])
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # make prediction for the with mask images
        prediction = loaded_model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Print the predicted class
        if predicted_class == 0:
            with_mask_pred.append(1)
        else:
            with_mask_pred.append(0)

    # Loop through the without mask images and make predictions
    for image_name in test_without_mask_images:

        # Load the image and convert it to a NumPy array
        image = load_img(os.path.join(test_path, "WithoutMask", image_name), target_size=input_shape[:2])
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # make predictions for the without mask images
        prediction = loaded_model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Print the predicted class
        if predicted_class == 0:
            without_mask_pred.append(1)
        else:
            without_mask_pred.append(0)
    
    # initialize the tp, tn, fp, fn
    tp, tn, fp, fn = 0, 0, 0, 0

    # calculate tp, fn through the with mask images predicitons
    for i in with_mask_pred:
        if i == 1:
            tp += 1
        else:
            fn += 1

    # calculate tn, fp throught the without mask images predictions
    for i in without_mask_pred:
        if i == 0:
            tn += 1
        else:
            fp += 1

    # calculate the precision, recall, and f1-score by their equations
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = tp + tn / ( tp + tn + fp + fn )

    print("Precision: ", precision)
    print("Recall: ",recall )
    print("F1-score: ", f1_score)
    print("Accuracy: ", accuracy)

    # return the precision, recall, and f1-score
    return precision, recall, f1_score, accuracy

evaluate()
