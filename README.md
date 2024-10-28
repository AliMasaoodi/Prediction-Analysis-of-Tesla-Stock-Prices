# Tesla Stock Price Prediction using Machine Learning
<h3>Prediction and Analysis of Tesla Stock Prices</h3>
This repository contains code for predicting the stock prices of Tesla using a Long Short-Term Memory (LSTM) model. The LSTM model is implemented using PyTorch, a popular deep learning framework.

## 🌐 Dataset
  The stock price data for Tesla is stored in the `tesla_stock_data.csv` file. The data includes the date and the closing prices of Tesla's stock. 

I used MinMaxScaler from scikit-learn normalizing the prices of the dataset.

## 📊 Prediction Model
The historical close prices of Tesla stock are plotted using matplotlib. The LSTM model is trained for a specified number of epochs with a defined learning rate. The loss is printed during training to monitor the model's convergence.
The predictions made by the trained model are inverse transformed to obtain actual stock prices. The RMSE is calculated to evaluate the model's accuracy in predicting both the training and test datasets.

## 📈 Prediction Results
Results shown below indicate the accurate prediction for future behaviour of the stock.

<img src="https://github.com/AliMasaoodi/Prediction-Analysis-of-Tesla-Stock-Prices/blob/main/figures/Tesla%20Stock%20Prediction%20-%20Predictions%20vs%20True%20Prices%20-%20Ali%20Masaoodi.jpg" alt="Tesla Stock Prediction - Predictions vs True Prices">
<img src="https://github.com/AliMasaoodi/Prediction-Analysis-of-Tesla-Stock-Prices/blob/main/figures/Tesla%20Stock%20Prediction%20-%20Historical%20Close%20Prices%20-%20Ali%20Masaoodi.jpg?raw=true" alt="Tesla Stock Prediction - Historical Close Prices">
<img src="https://github.com/AliMasaoodi/Prediction-Analysis-of-Tesla-Stock-Prices/blob/main/figures/Tesla%20Stock%20Prediction%20-%20Distribution%20of%20Close%20Prices%20-%20Ali%20Masaoodi.jpg" alt="Tesla Stock Prediction - Distribution of Close Prices">
<img src="https://github.com/AliMasaoodi/Prediction-Analysis-of-Tesla-Stock-Prices/blob/main/figures/Tesla%20Stock%20Prediction%20-%20Correlation%20Heatmap%20-%20Ali%20Masaoodi.jpg" alt="Tesla Stock Prediction - Correlation Heatmap">
<img src="https://github.com/AliMasaoodi/Prediction-Analysis-of-Tesla-Stock-Prices/blob/main/figures/Tesla%20Stock%20Prediction%20-%20Predictions%20vs%20Actual%20Prices%20-%20Ali%20Masaoodi.jpg" alt="Tesla Stock Prediction - Predictions vs Actual Prices">


## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/).

## Tools Used
[1] PyTorch: https://pytorch.org/
[2] Scikit-learn: https://scikit-learn.org/
[3] Matplotlib: https://matplotlib.org/
[4] Seaborn: https://seaborn.pydata.org/

<h2><img src="https://c.tenor.com/k3VfwdRd6cEAAAAi/loading-load.gif" width="50px" height="50px" alt=""> About Me </h2>

#### To be updated with my new projects, [Follow Me](https://github.com/AliMasaoodi):

![Ali Masaoodi LinkedIn](https://user-images.githubusercontent.com/33722769/208019646-5b06a2bd-5f75-43e2-8399-9150fe88db39.png)
[@AliMasaoodi](https://www.linkedin.com/in/ali-masaoodi/)



![Ali Masaoodi Github](https://user-images.githubusercontent.com/33722769/208020361-395da81a-8222-41e5-9f60-e82a846fa4fd.png)
[@AliMasaoodi](https://github.com/AliMasaoodi)

