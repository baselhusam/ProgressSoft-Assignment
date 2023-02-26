# Approach:

<br>

The approach taken in this program is to calculate the statistics of a given image in terms of mean,
standard deviation, minimum and maximum gray values, image height and width, and the number of
pixels. To achieve this, the program first loads the image from the given path using the **Bitmap** class in
the **_System.Drawing_** namespace. Then, it calculates the gray value of each pixel in the image by
converting the RGB color of the pixel to its corresponding grayscale value. After that, it calculates the
sum, squared sum, minimum, and maximum gray values of all the pixels to derive the mean, variance,
and standard deviation. Finally, the program displays the calculated statistics on the console.

<br>

### The solution has two files:

<br>

- **_Image_Stats.cs_** : this program has the **_ImageStats_** class which has a function called
    **_CalculateStats_** that calculates the image statistics for a given image.
- **_Program.cs:_** this program is the main program that has the code for making an object of this 
class then call the ImageStats function and gives it the image path as an input for this function,
then shows the Statistics for the image.

<br>

# Justification for Approach:

<br>

The conversion of the RGB color of a pixel to its corresponding grayscale value through the mathematical
equation from MathWorks. The equation is **0.2989 * R + 0.5870 * G + 0.1140 * B** (The R, G, and B
correspond to the RGB channels in the image). The mean, variance, and standard deviation are
commonly used statistical measures that provide insight into the distribution of gray values in an image.
The minimum and maximum gray values provide information on the range of gray values in the image.
The image height and width and the number of pixels provide information about the size of the image.

<br>

# Evaluation & Test Results:

<br>

The program can be tested by providing different images as input and verifying that the calculated
statistics are accurate. **NOTE** : I only tested it on Windows.

<br>

# How to Run the Program:

<br>

To run the scripts run the following command on the CMD: `dotnet run <path for the image>`

Then the program will output the statistics for the image, and it will show the:

- **Mean**
- **Standard Deviation**
- **Minimum Gray Value**
- **Maximum Gray Value**
- **Image Height**
- **Image Width**
- **Number of Pixels**

<br>

# Example:

<br>

```
Mean: 142.
Standard deviation: 37.

Minimum gray value: 0
Maximum gray value: 255

Image Height: 2975
Image Width: 2082

Number of pixels: 6193950
```

<br>

# Future Enhancement:

<br>

Possible future enhancements for this program include adding the ability to process multiple images at
once and to display the image along with its statistics. Additionally, the program could be extended to
perform more complex image processing tasks such as edge detection, image filtering, and object
recognition.



