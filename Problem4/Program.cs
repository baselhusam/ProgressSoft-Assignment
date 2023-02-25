using System;

namespace ImageStatistics
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Please provide the path to the image as a command line argument.");
                return;
            }

            // Load image and calculate stats
            var img = new ImageStats();
            img.CalculateStats(args[0]);

            // Print stats
            Console.WriteLine($" \nMean: {Math.Round(img.Mean, 2)}");
            Console.WriteLine($"Standard deviation: {Math.Round(img.StdDev, 2)} \n");
            Console.WriteLine($"Minimum gray value: {img.MinGrayValue}");
            Console.WriteLine($"Maximum gray value: {img.MaxGrayValue} \n");
            Console.WriteLine($"Image Height: {img.img_height}");
            Console.WriteLine($"Image Width: {img.img_width} \n");
            Console.WriteLine($"Number of pixels: {img.numPixels} \n");
        }
    }
}
