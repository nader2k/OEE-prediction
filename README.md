# OEE-prediction
In this project, the Overall Equipment Effectiveness (OEE) prediction was carried out using DNN and LSTM methods.
The evaluation of the firm's production and planning was also conducted.
The models were implemented using the K-fold cross-validation technique to overcome the problem of over-fitting and validate the employed methods on the training set. 
The DNN and LSTM method incorporate various hyperparameters. It is essential to test a variety of hyperparameters on the models in order to have a high-accuracy model.
Whit the use of avarage learning curve and K-fold cross-validation, you can test the models each time you train them.
The LSTM models are hard to train on every data. You can just implement them on specefic datasets. in addition, their trainin time takes a lot of time and when you use K-fold cross-validation technique, it takes more time! However, they are more accurate than DNNs in some ways.
When you want to fit LSTM models on time series datasets, it is important to make your datasets 3 dimensional.
The models were tested and compared using RMSE, MAE, and R2. it is better to have R2 near 1 and low amount of RMSE and MAE.
