using Microsoft.ML; 
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;
using Keras.Models;

namespace Problem3
{
    public class FaceMaskClassifier
    {
        private const string MODEL_FILE = @"D:\Projects\ProgressSoft_Assignment\Problem2\tf_model.onnx";
        private const int IMAGE_SIZE = 224;
        private const int NUM_CHANNELS = 3;

        private readonly PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

        public FaceMaskClassifier()
        {
            // Create a new MLContext
            var mlContext = new MLContext();

            // Load the TensorFlow model as a transform
            var tensorFlowModel = Model.LoadModel(MODEL_FILE);
            // var tensorFlowModel = Sequential.LoadFromModelFilePath(MODEL_FILE);
            
            
            // Define the pipeline for the prediction engine
            var pipeline = mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: IMAGE_SIZE, imageHeight: IMAGE_SIZE, inputColumnName: nameof(ImageData.Image))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: true, offsetImage: 117));


            // Fit the pipeline to the data and convert to an ITransformer
            ITransformer transformer = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new ImageData[] { new ImageData() }));
                        
            // Create the prediction engine
            _predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(transformer);

        }

        public bool Predict(string imagePath)
        {
            // Load the image data
            var imageData = new ImageData { ImagePath = imagePath };

            // Make a prediction with the model
            var prediction = _predictionEngine.Predict(imageData);

            // Get the predicted class label
            var isWithMask = prediction.Prediction < 0.5f;

            return isWithMask;
        }

        private class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [ImageType(224, 224)]
            public Bitmap Image;
        }

        private class ImagePrediction
        {
            [ColumnName("dense_2/Sigmoid")]
            public float Prediction;
        }
    }
}

