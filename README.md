# Term Project for Machine Learning
### Team:
Vivian Duong, Emily Lim, Robert Panerio

### Summary:
The project is based on a tutorial competition from Kaggle called Bag of Words Meets Bags of Popcorn. The competition focuses on using Word2Vec for sentiment analysis and deep learning. 

Our project focuses on a part of the competition where we take a dataset and apply a Bag of Words to get accurate predictions whether the a review is thumbs up or thumbs down. 

	
### Libraries Used:
1. Sklearn
   1. sklearn.feature_extraction.text
   2. sklearn.base
   3. sklearn.pipeline
   4. sklearn.linear_model
   5. sklearn.ensemble
   6. sklearn.model_selection
2. pandas
3. numpy
4. sys
5. re

### Files included in the Zip file:
1. main.py
2. labeledTrainData.tsv
3. testData.tsv

### How to run the program:
1. python3 main.py

### Prompts
The user will first be asked to select what type of estimator they would like to use: 
1. logistic regressor (l)
2. random forest classifier (r)

Next they can pick what they would like to happen with the run 
1. Demo (d) is the version that will be used in class for the presentation. The user will
   be asked to enter in a detailed movie review and the program will guess (using training
   data from kaggle) whether they liked the movie or not. (~1 min)
3. Kaggle (k) this uses the given training and test sets for the kaggle competition.
   Outputs the a file of predictions that can be submitted to Kaggle (~1 min).
5. Train test (t) uses only the training data and creates a train test split to get an
   accuracy score without submitting to kaggle (~1 min).
7. Logistic regression with optimization grid (o) will run with optimizations using RandomSearchCV.
   NOTE: we have already populated the results of best_params in the other modes so running any other version
   will yield the optimized results (~30 - 60 min).
9. Random forest with optimization grid (o) will run random forest with optimizations using RandomSearchCV.
   NOTE: we have already populated the results of best_params in the other modes so running any other version
   will yield the optimized results (~35 min). 