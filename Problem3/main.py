from face_mask_classifier_py import FaceMaskClassifier
import argparse

def main():

    """
    This script takes an image file path and a model file path as arguments, creates a FaceMaskClassifier object 
    using the provided paths, and predicts whether the face in the image is wearing a mask or not. 
    """

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Face Mask Classifier')

    # Add the arguments
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')

    # Parse the arguments
    args = parser.parse_args()

    # Create the classifier object
    classifier = FaceMaskClassifier(args.img_path, args.model_path)

    # Make a prediction
    is_with_mask = classifier.predict()

    # Print the result
    if is_with_mask:
        print("Prediction: Wearing Mask")

    else:
        print("Prediction: NOT Wearing Mask")

if __name__ == '__main__':
    main()
