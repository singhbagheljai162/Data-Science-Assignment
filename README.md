Data Science with Python: An Introduction to Linear Regression, Model Evaluation, and User Input Prediction

Welcome! This repository contains three simple programs to get you started with data science using Python. We’ll go over:

1.Exploring the Iris Dataset 2.Splitting Data for Training and Testing 3.Predicting Salary Using Linear Regression

Each program is designed to introduce you to essential concepts in data science and machine learning, like loading data, splitting it into training and testing sets, training a model, evaluating its performance, and making predictions.

Exploring the Iris Dataset In this program, we work with the famous Iris dataset, a collection of flower measurements that data scientists often use as a beginner’s dataset.
What does this program do?

Loads the Iris dataset. Displays the first five rows to give you a quick look at the data. Shows the dataset’s shape (number of rows and columns). Provides summary statistics, like the mean and standard deviation, which give insights into each feature. Why is this useful? Exploring the data is a crucial first step in any data science project. This helps us understand the data we’re working with, identify patterns, and spot potential issues (like missing values).

How to use it: Just run the code, and it will print out details of the Iris dataset for you to explore.

2.Splitting Data for Training and Testing The next step is to prepare the data so that we can build and test a model. This program shows how to split a dataset into a training set and a test set.
What does this program do?

Loads the Iris dataset (or you can replace it with other data if you want). Splits the data into training (80%) and testing (20%) sets. Prints the number of samples in both sets. Why is this useful? When building machine learning models, we want to make sure they can generalize to new data. Splitting data into training and testing sets allows us to test our model on data it hasn’t seen before, helping us evaluate how well it might perform in the real world.

How to use it: Run the code, and it will automatically split the dataset and show the number of samples in the training and testing sets.

3.Predicting Salary Using Linear Regression In the third program, we move into linear regression—a fundamental machine learning technique for making predictions. We’ll use it to predict a person’s salary based on their years of experience.
What does this program do?

Loads a dataset from a CSV file containing "YearsExperience" and "Salary" columns (you’ll need to provide the CSV file). Splits the data into training and testing sets. Trains a linear regression model using the training data. Evaluates the model using Mean Squared Error (MSE), which tells us how accurate the model is. Allows you to enter your own years of experience to get a predicted salary. Why is this useful? Linear regression is a powerful tool for understanding relationships between variables. In this case, we see how salary might increase as years of experience increases. By allowing user input, this program can provide personalized predictions, making it more interactive and engaging.

How to use it:

Ensure you have a CSV file with "YearsExperience" and "Salary" columns. Run the program to train the model and evaluate its accuracy. When prompted, enter a number representing years of experience, and the program will predict the expected salary. Prerequisites To run these programs, you’ll need:

Python (version 3.6 or later) pandas, scikit-learn, and numpy libraries Install these libraries if you don’t have them by running:

bash Copy code pip install pandas scikit-learn numpy Running the Programs After setting up your environment, you can run each program individually. For the third program, remember to update the CSV file path to match your own file’s location.

Conclusion These three programs are designed to guide you through basic data science steps: understanding your data, preparing it for analysis, building a model, evaluating it, and making predictions. Whether you're new to data science or looking to solidify your foundational skills, these examples offer a practical introduction.
