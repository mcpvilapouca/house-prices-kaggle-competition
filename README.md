# House Prices Kaggle Competition

This work was developed for the Kaggle Competition [House-Prices Advanced Regression Tecnhiques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

The aim is to predict the house prices using the <ins>Ames Housing Dataset</ins>. A training and test set are provided.

<img width="862" alt="housing-competition" src="https://user-images.githubusercontent.com/95075305/176886462-d891a78a-8b7b-497a-a14e-983d79105e67.png">

#### Data Preprocess
###### Replace NaN from train and test set, encode categorical features, remove outliers and save processed data to pickle
- preprocess_data.ipynb

#### Feature Selection
###### Read processed data from pickle. Select relevant features based on the correlation between them and with target. Save relevant features to pickle
- select_features.ipynb

#### Create Models
###### Script with the definition of 6 ML models used in the model selection
- create_models.py

#### Model Selection
###### Read processed data and relevant features from pickle. Read models from Create_Models.py. Evaluate models (default parameters) using RepeatedKFold. Allowed to choose the two most promising. Note that since the evaluation is performed between the logarithm of the predicted value and the logarithm of the observed sales price, we trained the model using the logarithm (np.log) of the target values.
- select_models.ipynb

#### Model Tunning
###### Read processed data and relevant features from pickle. Split the training set into training (90%) and validation sets (10%). Tune the two best models with RandomizedSearchCV using the splitted training set. Evaluate the tuned models with the validation set. Choose the best one and make predictions on the test set. Transform the predictions back to normal scale (np.exp). Create submission file
- tune_and_predict.ipynb

---------------------------------------------------------
###### This version gave me a score of 0.12657 (top 20%)
