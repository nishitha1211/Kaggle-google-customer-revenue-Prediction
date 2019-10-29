# KaggleProject
Google Analytics Customer Revenue Prediction
The URL for the competition: https://www.kaggle.com/c/ga-customer-revenue-prediction

The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

In this competition, we are challenged to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer.

We are predicting the natural log of the sum of all transactions per user. Once the data is updated, as noted above, this will be for all users in test_v2.csv from December 1st, 2018 to January 31st, 2019. 
For every user in the test set, the target is:
y_user=∑i=1ntransactionuseri
targetuser=ln(yuser+1)
