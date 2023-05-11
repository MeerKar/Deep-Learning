# Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding 
with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, 
you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will 
be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 
organizations that have received funding from Alphabet Soup over the years. 
Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns

APPLICATION_TYPE—Alphabet Soup application type

AFFILIATION—Affiliated sector of industry

CLASSIFICATION—Government organization classification

USE_CASE—Use case for funding

ORGANIZATION—Organization type

STATUS—Active status

INCOME_AMT—Income classification

SPECIAL_CONSIDERATIONS—Special considerations for application

ASK_AMT—Funding amount requested

IS_SUCCESSFUL—Was the money used effectively


## Step 1: Preprocess the Data

Using Pandas and scikit-learn’s StandardScaler(), the dataset has been preprocessed. This step prepares 
for Step 2, where the compile, train, and evaluate the neural network model happens.


By using Google Colab, 
Read in the charity_data.csv to a Pandas DataFrame, and identified the following in the dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

<img width="1016" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/84b8a1d2-1faa-47d8-abeb-f9803c66b602">

Determine the number of unique values for each column.

<img width="309" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/3b602178-b5ac-4d2c-9f2b-c32b7ba91c35">


For columns that have more than 10 unique values, determine the number of data points for each unique value.


Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value,
Other, and then check if the binning was successful.

<img width="440" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/4acf5f33-6f2d-4636-a0bd-754e956f7185">

<img width="469" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/59809b07-3a98-4b0a-b35e-abb6e7059b5c">




Use pd.get_dummies() to encode categorical variables.

<img width="582" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/e7a111ed-5ade-4bf1-b785-af701cb8eebe">

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

<img width="673" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/82e1d1f8-e6c4-4670-9890-493a8d7944bc">


Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

<img width="413" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/5d321a94-2b57-40a7-9119-694af971e2a3">

## Step 2: Compile, Train, and Evaluate the Model

Using the knowledge of TensorFlow,  design a neural network, or deep learning model, to create a binary classification model 
that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. 
Need to think about how many inputs there are before determining the number of neurons and layers in the model. 
Once completed that step, then compile, train, and evaluate the binary classification model to calculate the model’s 
loss and accuracy.


Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

<img width="920" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/c331249b-7a4d-4278-8e59-13f0de623cba">

Create an output layer with an appropriate activation function.


Check the structure of the model.

Compile and train the model.

<img width="866" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/573e1c4c-5ea3-4ace-b67b-e65dd2cae8fd">

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

<img width="671" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/600ae98b-de07-467e-9632-7ee7dfb119d4">

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

<img width="465" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/e151a51e-5e6f-4c9f-94c1-c1c64fc8d1df">


## Step 3: Optimize the Model

Using knowledge of TensorFlow, optimize the model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

Dropping more or fewer columns.

Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.


Created a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Imported dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

<img width="748" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/0368749c-b5ce-4488-9655-1c166a4f791e">

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

<img width="583" alt="image" src="https://github.com/MeerKar/Deep-Learning/assets/116701851/8f5e5407-fb01-421c-83fc-715a98978d7b">

Step 4: Write a Report on the Neural Network Model



