import pandas as pd
import joblib


from preprocessing import DataPreprocessor

def load_model(model_path):
    """Load the trained xgb_regressor model from the specified path."""
    return joblib.load(model_path)

def preprocess_data(filename):
    """Preprocess new data using the DataPreprocessor class."""
    processor = DataPreprocessor(filename)
    numeric_df = processor.preprocess_data()
    return numeric_df


def make_predictions(model, new_data):
    """Make predictions using the loaded model and preprocessed new data."""
    predictions = model.predict(new_data)
    return predictions

def save_predictions(predictions, output_path):
    """Save the predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(
        output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    
    # Path to the properties.csv data (to be predicted)
    filename = "datasets/immo-eliza/properties.csv"


    # Path to the saved xgb_regressor model
    model_path = 'xgb_model.joblib'

    # Load the trained xgb_regressor model
    model = load_model(model_path)

    # Preprocess the new data
    new_data = preprocess_data(filename).drop(
        'price', axis=1, errors='ignore')

    # Make predictions on the new data
    predictions = make_predictions(model, new_data)

    # Save or display the predictions
    save_predictions(predictions, 'predictions_on_properties.csv')