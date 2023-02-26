# Approach:

<br> 

To predict the stock's most recent 20% prices, I used two approaches, one using Machine Learning and
the other using Deep Learning.

<br>

### The solution includes four files:

<br>

- **_TimeSeries_with_ML.ipynb_** : Jupyter notebook to solve the problem with Machine Learning
- **_TimeSeries_with_DL.ipynb_** : Jupyter notebook to solve the problem with Deep Learning
- **_prices.txt_** : the data file which has the dates and the prices for the stock in txt format.
- **_prices.csv_** : the data in CSV, the way of converting happened in the **_TimeSeries_with_ML.ipynb_**
    file.


<br>

# The Dataset:

<br>

The Data come in .txt format which has values in a comma-separated way. It has 2 features, the **date** ,
and the **stock prices**. It has **1259** rows (instances) of data. The data has values for each stock for each
day, and it starts from **2017 - 02 - 21** to **2022 - 02 - 17** , which is approximately **5 years**.

What I did is read this file in python, extract the values, and make them as Pandas data frame, make the
data type for the date feature as datetime through pandas (for manipulating the data as a date data
type), make the date as the index for this data frame, and save this data frame a CSV file using pandas
**_.to_csv_** method.

<br>

## Machine Learning Approach:

<br>

I first read the data from the prices.txt file, then extract the values needed and make a data frame from
it, then save it as a CSV file. Before everything, I changed the date datatype to datetime, for working with
it as a date, not as an object (string).

After that, I prepare the data for splitting. The splitting happened as the first **80%** of the data was the
training dataset and the final **20%** was the test dataset. This give that we have **1,007** instances for the
training data, and **252** instances for the testing data. This means that the data is **1259** instances (stock
prices).

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture10.png)

Then, I created a function called
**_create_features_** and take data frame as
input. This function creates features for
the data frame, the features are related
to the data and extract information from
it, such as the day of the week, quarter,
month, year, etc.

Then, do EDA (Exploratory Data Analysis)
through some plots of the data. After this
step, I split the data into features, and
the target (X, y) just like it is a supervised
learning approach.
<br>

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture11.png)


After that, build the model and train it on the data. The model I used is an **_XGBoost Regressor_** with a
1,000 estimator, 0.01 learning rate, max depth of 2, and evaluation metric as ‘MAE’ (Mean Absolute
Error). Then, evaluate the model with the MAE (Mean Absolute Error), and RMSE (Root Mean Squared
Error).

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture12.png)


Finally, plot the feature importances for the model and horizontal bar chart, and we conclude from it
that the **_year_** feature is the most important feature for the model.

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture13.png)


<br>

## The Deep Learning Approach:

<br>

I used a window method to preprocess the data, which involved creating windows of fixed size and
predicting the next value in each window. I then defined 3 main functions I will use in the notebook,
**_plot_series_** which visualizes time series data, **_windowed_dataset_** which generated dataset windows, and
**_model_forecast_** which used an input model to generate predictions on data windows.

I then split the data into training and testing sets. That training data is in the range of 40% to 90% of the
data, and the testing set has the last 20% of the data. I did this for the training set because the first 40%
of the data as almost similar to each other and the range of the price for them is 20 – 100, and recent
years have prices more than 500, so there is no benefit to train the model on a data its values kind of
disappeared. Note that there will be 10% will be overlapping with the training and testing sets, which
will not give us a real evaluation number in the evaluation phase.

I used the window method because it is a common technique to preprocess time series data for deep
learning models.

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture14.png)


After that, I used a deep learning model with two LSTM layers and multiple Dense layers to train the
model on the training set and then tested it on the testing set. I used an LSTM-based model because
LSTMs are known to perform well on time series data and can capture the temporal dependencies in the
data.

After building the model I trained it on the
training data just to pick the best value for the
learning rate for this model on this data. This
happened with the
**_callbacks.LearningRateScheduler_** , and then
pick the best value of the learning rate and
train the model with this value.

![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture15.png)


I chose MAE as the loss function, and the RMSE as the metrics. I choose them because they are popular
metrics in time-series problems, which can show the model performance in a good way.


Then, I made predictions on the train, and the test data, and plot the results.


![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture16.png)
![img](https://github.com/baselhusam/ProgressSoft-Assignment/blob/main/images/Picture17.png)

Clearly, we can see that the model is overfitted on the training data, I tried the change the model
architectures multiple times, but I couldn’t prevent the overfitting, I tried to change some values for
parameters like window size, batch size, and number of units (neurons) inside the layer, and many other
parameters.

_MAE & RMSE while Training_ (^) _Zoom in to the MAE & RMSE while Training
Model Performance on the Test set Model Performance on the Training set_^


<br>

# Evaluation:

<br>

The performance of the models was evaluated on the test dataset using MAE (Mean Absolute Error), and
RMSE (Root Mean Square Error) metrics. After applying the evaluation step to our models, I got the
following values for the metrics.

<br>

| **Model \ Metrics**         | **MAE**     | **RMSE** |
|--------------|-----------|------------|
| **Machine Learning** | 118.24      | 166.59        |
| **Deep Learning**      | 93.81  | 115.28       |

<br>

As we can see that the deep learning model performed better than the machine learning model, maybe
that is because of the complexity of this model, which has multiple LSTMs and Dense layers. The DL
model has **150,751 parameters.**

<br>

# Python Version:

<br>

The script was developed using Python **_3.9.14_**

<br>

# How to Run the Scripts:

<br>

For both the ML and DL notebooks, _write_ **_pip install - r requirements.txt_** in **CMD** , then open them and
execute the cells from above to bottom, and if you want to explore different values for the parameters of
the models, notice that the prices.txt file should be in the same directory of the notebooks.

**NOTE:** I only tested it on Windows.

<br>

# Future Enhancement:

<br>

For the machine learning approach, future enhancements could include trying out more regression
models and hyperparameter tuning to improve the model's performance. For the deep learning
approach, enhancements could include trying out different architectures and hyperparameter tuning to
improve the model's performance. Additionally, feature engineering could also be explored to add more
relevant features to the dataset.


