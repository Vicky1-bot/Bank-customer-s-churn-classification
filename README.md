# Bank-customer's-churn-classification
An Artificial Neural Network or a Deep Learning Model that identifies whether the customer will leave the bank or not.

### Introduction

This is an international Bank with millions of customers spread all around Europe mainly in, three countries, Spain, France and Germany. In the last six months the bank detected that the churn rates started to increase when compared to the average rate, so they decided to take measures. The bank decided to take a small sample of 10,000 of their customers and retrieve some information.

For six months they followed the behaviour of these 10,000 customers and analysed which stayed and who left the bank. Therefore, they want us to develop a model that can measure the probability of a customer leaving the bank.

Our goal for this task is to create a model to tell the bank which of the customers are at higher risk to leave.

### Frame the problem

Before looking at the data it’s important to understand how does the bank expect to use and benefit from this model? This first brainstorming helps to determine how to frame the problem, what algorithms to select and measure the performance of each one.

We can categorize our Machine Learning (ML) system as:

Supervised Learning task: we are given labeled training data (e.g. we already know which customers left);

Classification task: our algorithm is expected to assign a binary value to each client indicating the probability of him leaving or staying with the bank.

Plain batch learning: since there is no continuous flow of data coming into our system, there is no particular need to adjust to changing data rapidly, and the data is small enough to fit in memory, so plain batch learning should work.

### Select a Performance Measure
### Available data
This dataset is composed of 10,000 rows representing a customer and 11 features.

 #### 1.Exploring the dataset:
 
   We will start by importing the necessary libraries. For this first step, we’ll mainly work with Pandas, Numpy and Matplotlib.As you can see, the dataset is very small in terms of features available, with the majority being numeric and only three categorical.Looking at the data we can see there are two categorical features that need to be encoded : Gender and Geography .There appear to be no NaN values in our dataset, which is good.
   
 #### 2.Feature engineering: 
 
  Let's Converting categorical features into numerical features using get_dummies().here basically,Converting 'Geography' and 'Gender' column and appending the columns to the dataframe.
 
 #### 3.Data preprocessing:
 • Split the dataset into independent features (ie: X) and label (ie: y).

 • Split the dataset further into train and test sets.

 • Apply feature scaling to train and test sets of independent features.
 
 #### 4.Building the ANN(Artificial neural network) and Model evaluation:
 So, now that you’re more instructed on ANN here is a summary of all the steps you need to take to build and train an Artificial Neural Network! Let’s refresh our memories!
 '''python
    #Import Keras library and packages
    import keras
    import sys
    from keras.models import Sequential #to initialize NN
    from keras.layers import Dense #used to create layers in NN
 '''


Step 1. Randomly initialise the weights with small numbers close to zero but not zero. This will be done by our Dense function.

Step 2. Distribute features of the first observation, from your dataset, per each node in the input layer. Thus, eleven independent variables will be added to our input layer.

'''python
   #Initialising the ANN - Defining as a sequence of layers or a Graph
   classifier = Sequential()
'''
Adding the input layer

units - number of nodes to add to the hidden layer.

Tip: units should be the average of nodes in the input layer (11 nodes) and the number of nodes in the output layer (1 node). For this case is 11+1/2 = 6

kernel_initializer - randomly initialize the weight with small numbers close to zero, according to uniform distribution.

activation - Activation function.

input_dim - number of nodes in the input layer, that our hidden layer should be expecting

'''python
   #Input Layer
   classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11 ))
'''

Step 3. Foward-Propagation : 
From the input to the output the neurons are activated, and the impact they have in the predicted results is measured by the assigned weights. Depending on the number of hidden layers, the system propagates the activation until getting the predicted result y.

To define the first hidden layer, we firstly will have to define an activation function. The best one is the Rectifier Function and we’ll choose this one for the hidden layers. Furthermore, also by using a Sigmoid function to the output layer will allow us to calculate the probabilities of the different class (leaving or staying the bank). In the end, we will be able to rank the customers by their probability to leave the bank.

Adding Second hidden layer

There is no need to specify the input dimensions since our network already knows.

'''python
   classifier.add(Dense(units = 6, kernel_initializer = ‘he_uniform’, activation = ‘relu’))
'''
Adding Output layer

units — one node in the output layer

activation — If there are more than two categories in the output we would use the softmax

'''python
classifier.add(Dense(units = 1, kernel_initializer = ‘glorot_uniform’, activation = ‘sigmoid’))
'''

Step 4. Cost Function : Measure the generated error by comparing the predicted value with the true value.

Stochastic Gradient Descent — Compiling the ANN

optimizer — algorithm to use to find the best weights that will make our system powerful

loss — Loss function within our optimizer algorithm

metric — criteria to evaluate the model

'''python
classifier.compile(optimizer = ‘adam’,loss= “binary_crossentropy”,metrics=[“accuracy”])
'''
Step 5. Back-Propagation: from the output to the input layer, the calculated error is back-propagated and the weights get updated according to the influence they had on the error. The learning rates indicate how much these weights are updated.

Step 6. Reinforcement Learning : Update weights at each observation (steps 1 to 5) or Batch Learning : Update the weights after each batch of observations (steps 1 to 5)

Step 7. When the system has gone through the whole training dataset, an epoch has been run. Redo more epochs.

Fitting the ANN to the Training Set

batch_size : number of observations after which we update the weights

epochs: How many times you train your model
'''python
   model_history = classifier.fit(X_train, y_train, batch_size=10, validation_split=0.33, epochs=100)
'''   

<img width="872" alt="image" src="https://user-images.githubusercontent.com/76062756/148526056-8b75759d-9c60-4226-bfd3-97e3e9a9888f.png">

This is our trained ANN model which, after running 100 epochs on the training set, returned an accuracy of around 86%
 
 



 
 
 
 
 

