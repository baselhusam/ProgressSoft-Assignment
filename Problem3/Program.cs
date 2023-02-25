using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Problem3
{
    class Program
    {
        static void Main(string[] args)
        {

            // Create an instance of the FaceMaskClassifier class
            FaceMaskClassifier classifier = new FaceMaskClassifier();

            // Get the path of the image to classify
            string imagePath = @"D:\Projects\ProgressSoft_Assignment\Problem2\Face Mask Dataset\Test\WithMask\3.png";

            // Make a prediction using the classifier
            bool isWithMask = classifier.Predict(imagePath);

            // Print the result
            if (isWithMask)
            {
                Console.WriteLine("The person in the image is wearing a face mask.");
            }
            else
            {
                Console.WriteLine("The person in the image is not wearing a face mask.");
            }

        }
 
    }
}