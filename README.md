# immo-eliza-ml 🏠
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Pandas](https://img.shields.io/badge/uses-Pandas-blue.svg)
![Matplotlib](https://img.shields.io/badge/uses-Matplotlib-blue.svg)
![Plotly](https://img.shields.io/badge/uses-Plotly-ff69b4.svg)
![Scikit-learn](https://img.shields.io/badge/uses-Scikit--learn-orange.svg)
![Joblib](https://img.shields.io/badge/uses-Joblib-red.svg)

## 🏢 Description
The real estate company Immo Eliza asked to create a machine learning model to predict prices of real estate properties in Belgium.

After the scraping, cleaning and analyzing, this script do preprocessing data and finally build a performant machine learning Model training and Model evaluation to be used for price prediction 

## Repo structure
```
├── datasetes/immo-eliza
│   ├── properties.csv
├── ModelingData/
│   ├── PreprocessAndRegression.ipynb
├── .gitignore
├── predict.py
├── predictions_on_properties.csv [Output file]
├── preprocessing.py  
├── README.md
├── requirements.txt
└── train.py
└── xgb_model.joblib [Saved model]
```


## 🛎️ Usage

1. Clone the repository to your local machine.

    https://github.com/soha-ms/immo-eliza-ml

2. Navigate to the project directory and install the required dependencies:

    pip install -r requirements.txt

3 .To run the script, you can execute the `train.py` file from your command line:
The script reads data from properties.csv file, and do cleaning and preprocessing for numerical and categorical feature then split train model and use best parameter for *xgboost* model to get best score and error meteric values. This model file is saved as 'xgb_model.joblib' to be used in predict.py

    train.py   
 
```python
# Call all functions and do preprocessing data
# by calling class DataPreprocessor from preprocessing.py 
def main():    
   
    processor = DataPreprocessor(filename)
    numeric_df = processor.preprocess_data()
    
# Split data for modeling
def train_test_data(numeric_df):

# Scaler data
def normalize_data(X_train, X_test):   

# XGBRegressor
# Fit the model on the training data
# Calculate train and test scores
def train_model(X_train, X_test, y_train, y_test, **best_params): 

# Optimize the model's performance
def tuning_param(X_train, y_train): 
```

4. Then run this file which is used to open saved model 'xgb_model.joblib' and predict price. Then save it as 'predictions_on_properties.csv'

    predict.py   

```python
   def load_model(model_path):
   def preprocess_data(filename):
   def make_predictions(model, new_data):
   def save_predictions(predictions, output_path):
```

5. This file is called by both train.py and predict.py to preprocess data before prediction  

    preprocessing.py  

## ⏱️ Timeline

This project took 5 days for completion.

## 📌 Personal Situation
This project was done as part of the AI Boocamp at BeCode.org. 

Connect with me on [LinkedIn](https://www.linkedin.com/in/soha-mohamad-382b44219/).