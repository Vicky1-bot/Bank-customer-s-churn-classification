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

