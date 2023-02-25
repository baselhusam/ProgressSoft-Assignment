using System;
using System.Drawing;

namespace ImageStatistics
{
    class ImageStats
    {
        // Properties to hold calculated values
        public double Mean { get; private set; }
        public double StdDev { get; private set; }
        public int MinGrayValue { get; private set; }
        public int MaxGrayValue { get; private set; }
        public int numPixels {get; private set; }
        public int img_height {get; private set; }
        public int img_width {get; private set; }

        // Method to calculate image statistics
        public void CalculateStats(string imagePath)
        {
            // Load the image
            Bitmap image = new Bitmap(imagePath);

            // Calculate the image statistics
            double sum = 0;
            double sumSquared = 0;
            MinGrayValue = 255;
            MaxGrayValue = 0;

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    // Get the color of the pixel
                    Color pixel = image.GetPixel(x, y);

                    // Convert the pixel color to grayscale value
                    int grayValue = (int)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);

                    // Update sum and squared sum for mean and variance calculations
                    sum += grayValue;
                    sumSquared += grayValue * grayValue;

                    // Update minimum and maximum values
                    if (grayValue < MinGrayValue){MinGrayValue = grayValue;}
                    if (grayValue > MaxGrayValue){MaxGrayValue = grayValue;}
                }
            }

            // Calculate the mean and standard deviation
            img_width = image.Width;
            img_height = image.Height;
            numPixels = image.Width * image.Height;
            Mean = sum / numPixels;
            double variance = sumSquared / numPixels - Mean * Mean;
            StdDev = Math.Sqrt(variance);
        }
    }
}


