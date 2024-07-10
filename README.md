 # Where's Nifty ?

## **Introduction :**

In this project, we will be predicting the high and low prices of Nifty50 index using a Stacked LSTM neural network. 

The NIFTY 50 is a benchmark Indian stock market index that represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange.
<br>**High** is the highest price at which the scrip has traded during a trade session.
<br>**Low** is the lowest price at which the scrip has traded during a trade session.

The model is trained on historical price data to predict future prices. The LSTM model is chosen for its ability to handle time series data and capture temporal dependencies.

## **Reading the libraries:**

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/46ab76ca-c6e7-4cc4-b7b7-846dc4b24375)

Pandas is a library that is used for data analysis. It also acts as a wrapper over Matplotlib and NumPy libraries. For instance, .plot() from Pandas performs a similar operation to Matplotlib plot operations.

NumPy is used for performing a wide variety of mathematical operations for arrays and matrices. In addition to this, the steps taken for computation from the NumPy library are effective and time-efficient.

Matplotlib is a low level graph plotting library in python that serves as a visualization utility. It creates publication quality plots and makes interactive figures that can zoom, pan, update.

## **Dataset:**

The dataset is used in this project contains historical Nifty stock prices. It includes various columns such as High, Low, Open, Close, Date, Shares Traded and Turnover.

#### Note: The data was taken from 
https://www.nseindia.com/reports-indices-historical-index-data

## **Data Preprocessing:**

**Loading Data:** The data is loaded using pandas.

**Checking for Missing Values:** There are no missing values and duplicate rows in our dataset. So there is no room for that error rectification.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/5a99072d-1ca5-4a8d-93aa-fc40080934aa)

**Scaling Data:** The High and Low columns are normalized using MinMaxScaler to bring values within the range of 0 to 1.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/1e50c00c-d22a-4a44-b57b-8ecd98f8a67d)


## **Data Exploration and Visualization:**  

We will visualize the High column (in y-axis) of the dataset by plotting it against days (in x-axis).

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/6ade5d7b-0d35-4c07-af7f-6ddfc409e829)

Next up is the visualization of the Low column.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/3c332c91-283f-4dc2-93c9-64e1cfb456f6)

## **Prediction of the High Price :**

### **Splitting data set into Training Data and Test Data:**

It is important to divide the data into training and test set. The training set is used to train the machine learning models under consideration. After successfully training the models, we take a look at their performance over the test set where the output is already known. After the machine learning models predict the outcome for the test set, we take those values and compare them with the known test outputs to evaluate the performance of our model.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/b2e06e62-915b-49b9-8530-c6f1f599dcf3)

70% of the dataset is now the training set and remaining 30% is the test set.

### **Creating smaller datasets:**

We will now convert our dataset into dependent and independent feature based on the timesteps (i.e., 100 over here).

Suppose we have **training set values** as - 120,130,125,140,134,150 and **test set values** as - 160,190,154,166.
<br> timesteps=3

then, 
<br>f1 = 120, f2 = 130, f3 = 125, o/p = 140 (in 1st round)
<br>f1 = 130, f2 = 125, f3 = 140, o/p = 134 (in 2nd round)
<br>f1 = 125, f2 = 140, f3 = 134, o/p = 150 (in 3rd round)

the f1, f2 and f3 belong to X_train and o/p belongs to X_test.

Same happens with the test set.
 
![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/6f48d29d-b0eb-4d3b-af5d-ca8921996c0b)

Next we will convert the shape of X-train and X_test to 3 dimensions so that we can give it as an input to our LSTM model.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/8898d295-b12a-4337-acd9-b08c4fef5652)


### **Model Building:**

An LSTM model is constructed using Keras. The model consists of 3 LSTM layers with dropout to prevent overfitting and 2 dense layers for the output.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/35d6cd51-262a-44c4-a683-5b7258bbf216)

Now we fill fit X_train and Y-train in our model

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/1f9b5a99-b38b-4a20-84ac-7bc5d623b4f0)

Epoch refers to the one entire passing of training data through the algorithm. The training data is broken down into small batches to overcome the issue of storage space limitations of a computer system. These smaller batches can be easily fed into the machine learning model to train it. 

Next up is the model prediction.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/482a533b-656a-49ce-ac2c-4ca0c3326f13)

### **Model Evaluation:**

We will first inverse the normalization of the predicted values train and test data and then plot the graph to evaluate its performance.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/41240709-ae95-45bb-8212-a9f4a4363cad)

The blue color output is the complete dataset. The orange color output is the training data prediction and the green color is the test data prediction.
<br> Our stacked LSTM model has performed quite well as the predicted plots are very close to the original ones.

### **Predicting the next 7 days Nifty50 index High Price:**

We will take last 100 days data of High column and reshape it into (1,-1) and store it in a list.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/0407a819-c399-4c1c-b03c-651ccdb1d2ff)


Then we are passing the list to our stcked LSTM model and predicting. The predicted value is stored in the yhat variable and adding that yhat variable inside the final output and also adding it to the previous input, i.e. the temp_input.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/57d91302-b4a0-4808-900b-4c7ae8f43a72)

Finally we plot the predicted 7 days values in a graph.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/71c97e15-2d66-4f89-ba28-3d489e53bf62)

Our next 7 days prediction is:
<br>Day 1 - 22713.92924082
    <br>Day 2 - 22710.17390293
       <br>Day 3 - 22687.02057463
       <br>Day 4 - 22656.65483176
       <br>Day 5 - 22626.06883446
       <br>Day 6 - 22598.36908781
       <br>Day 7 - 22574.39622951

## **Prediction of the Low Price :**

The low index price of Nifty50 is calculated in the same manner as high.
<br>Just a small change is needed.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/187a9620-ee23-47d2-87ed-8608ce0b231a)

Below is the performance of our stacked LSTM on low data.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/80d5a4b6-c3d9-45b5-8190-11967b49c299)

Next is the prediction graph.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/fa354f70-dab7-4382-9ee5-76dc50c75b97)

Our next 7 days prediction is:
<br>Day 1 - 22508.54107529
       <br> Day 2 - 22464.60766007
       <br> Day 3 - 22385.70151366
       <br> Day 4 - 22297.22164174
       <br> Day 5 - 22213.46989807
       <br> Day 6 - 22139.79846507
       <br> Day 7 - 22076.71209247




