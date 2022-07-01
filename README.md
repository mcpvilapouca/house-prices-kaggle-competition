# House Prices Kaggle Competition

<img width="862" alt="housing-competition" src="https://user-images.githubusercontent.com/95075305/176886462-d891a78a-8b7b-497a-a14e-983d79105e67.png">

- This work was developed for the Kaggle Competition [House-Prices Advanced Regression Tecnhiques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

The aim is to predict the house prices using the <ins>Ames Housing Dataset</ins>. A training and test set are provided.

#### Data Preprocess
- preprocess_data.ipynb

###### Replace NaN from train and test set, encode categorical features, remove outliers and save processed data to pickle

#### Feature Selection
- select_features.ipynb

###### Read processed data from pickle. Select relevant features based on the corrrelation between them and with target. Save relevant features to pickle

#### Create Models
- create_models.py

###### Script with the definition of 6 ML models used in the model selection

#### Model Selection
- select_models.ipynb

###### Read processed data and relevant features from pickle. Read models from Create_Models.py. Evaluate models (default parameters) using RepeatedKFold. Allowed to choose the two most promising. 

#### Model Tunning
- tune_and_predict.ipynb

###### Read processed data and relevant features from pickle. Split the training set into training (90%) and validation sets (10%). Tune the two best models with RandomizedSearchCV using the splitted training set. Evaluate the tuned models with the validation set. Choose the best one and make predictions on the test set. Create submission file

**This version gave me a score of 0.12657 (top 20%)**
