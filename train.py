import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import joblib

from preprocessing import PreprocessorData
 
def show_heatmap(numeric_df):
    corr_matrix = numeric_df.corr(method='spearman')
    plt.figure(figsize=(15, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Diabetes Dataset")
    plt.show()

def train_test_data(numeric_df):
    X = numeric_df.drop(columns=['price'])
    y = numeric_df['price']
    # Splitting data into train and test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2)
    print(X_train.shape)
    return X_train, X_test, y_train, y_test 

### Normalization
def normalize_data(X_train, X_test):    
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test

def train_model(X_train, X_test, y_train, y_test, **best_params):


    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state = 42 ,**best_params)
                                     
                                     #max_depth=5, n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the model on the training data
    xgb_regressor.fit(X_train, y_train)

    # Calculate train and test scores
    train_score = xgb_regressor.score(X_train, y_train)
    test_score = xgb_regressor.score(X_test, y_test)
    print(f"Training R² score: {train_score:.2f}")
    print(f"Testing R² score: {test_score:.2f}")


    # Predictions on the test set
    y_pred = xgb_regressor.predict(X_test)   


    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Save the model
    joblib.dump(xgb_regressor, 'Model/xgb_model.joblib')

#Optimize the model's performance
def tuning_param(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Display the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    return best_params

def main():    
    filename = "datasets/immo-eliza/properties.csv"      
    processor = PreprocessorData(filename)
    numeric_df = processor.preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_data(numeric_df)       
    train_model(X_train, X_test, y_train, y_test)

    ### Normalization and test score again
    X_train, X_test = normalize_data(X_train, X_test)

    #best_params = tuning_param(X_train, y_train)
    best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1}
    print('\nScores after optimizing the model performance using GridSearchCV() : ')
    train_model(X_train, X_test, y_train, y_test, **best_params) 

if __name__ == '__main__':
    main()   