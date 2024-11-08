import pandas as pd
import joblib


from preprocessing import PreprocessorData

def load_model(model_path):
    #Load the trained xgb_regressor model
    return joblib.load(model_path)

def preprocess_data(filename):
    #Preprocess data using the PreprocessorData class.
    processor = PreprocessorData(filename)
    numeric_df = processor.preprocess_data()
    return numeric_df


def make_predictions(model, new_data):
    #Make predictions using the loaded model
    predictions = model.predict(new_data)
    return predictions

def save_predictions(predictions, output_path):
    #Save the predictions to a CSV file."
    pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(
        output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    
    filename = "datasets/immo-eliza/properties.csv"

    # Path to the saved xgb_regressor model
    model_path = 'Model/xgb_model.joblib'

    # Load the trained xgb_regressor model and Preprocess
    model = load_model(model_path)
    new_data = preprocess_data(filename).drop('price', axis=1, errors='ignore')

    # Make predictions on the new data and  Save predictions
    predictions = make_predictions(model, new_data)    
    save_predictions(predictions, 'Output/predictions_on_properties.csv')