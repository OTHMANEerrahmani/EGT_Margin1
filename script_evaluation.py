# 0. IMPORTS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split  # For typical ML, but we'll do a chronological split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

# --- Configuration ---
TARGET_COLUMN = 'EGT Margin'
DATE_COLUMN = 'Flight DateTime'
CSN_COLUMN = 'CSN'
BASE_FEATURES = ['Vibration of the core', CSN_COLUMN, 'Cycles Since last shop visit']

# --- 1. LOAD DATA ---
def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    Args:
        file_path: Chemin du fichier CSV ou objet de fichier uploadé
    Returns:
        DataFrame pandas ou None en cas d'erreur
    """
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# --- 2. PROCESS DATA ---
def process_data(df):
    """Processes the DataFrame: type conversion, sorting, feature engineering."""
    if df is None:
        return None

    print("\n--- Starting Data Processing ---")

    # a. Convert 'Flight DateTime' to datetime objects
    if DATE_COLUMN not in df.columns:
        print(f"ERROR: Date column '{DATE_COLUMN}' not found.")
        return None
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    print(f"'{DATE_COLUMN}' converted to datetime.")

    # b. Handle potential comma decimal separators and convert to numeric
    # (EGT Margin is the target, other features are in BASE_FEATURES)
    cols_to_convert_numeric = [TARGET_COLUMN] + BASE_FEATURES
    for col in cols_to_convert_numeric:
        if col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                print(f"Converting column '{col}' from object/string to numeric...")
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
            elif not pd.api.types.is_numeric_dtype(df[col]):  # If not object but also not numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors to NaN
            print(f"Column '{col}' is now numeric (dtype: {df[col].dtype}).")
        else:
            print(f"Warning: Expected column '{col}' not found in DataFrame.")

    # c. Sort data chronologically
    df = df.sort_values(by=[DATE_COLUMN, CSN_COLUMN]).reset_index(drop=True)
    print("DataFrame sorted by DATE_COLUMN and CSN_COLUMN.")

    # d. Feature Engineering
    print("Starting feature engineering...")
    engineered_features = []

    # Lags for Target and key features
    lags_target = [1, 2, 3, 5, 7]  # How many steps back to look
    lags_features = [1, 2, 3]

    for lag in lags_target:
        new_col_name = f'{TARGET_COLUMN}_lag{lag}'
        df[new_col_name] = df[TARGET_COLUMN].shift(lag)
        engineered_features.append(new_col_name)

    for feature_col in ['Vibration of the core']:  # Add other features if desired
        if feature_col in df.columns:
            for lag in lags_features:
                new_col_name = f'{feature_col}_lag{lag}'
                df[new_col_name] = df[feature_col].shift(lag)
                engineered_features.append(new_col_name)
        else:
            print(f"Warning: Column '{feature_col}' not found for lag feature engineering.")

    # Rolling window statistics for Target and key features
    window_sizes = [5, 10]  # Example window sizes
    for window in window_sizes:
        # Target rolling stats
        new_col_mean = f'{TARGET_COLUMN}_roll_mean{window}'
        new_col_std = f'{TARGET_COLUMN}_roll_std{window}'
        df[new_col_mean] = df[TARGET_COLUMN].rolling(window=window).mean().shift(
            1)  # use .shift(1) to prevent data leakage
        df[new_col_std] = df[TARGET_COLUMN].rolling(window=window).std().shift(1)
        engineered_features.extend([new_col_mean, new_col_std])

        # Feature rolling stats
        for feature_col in ['Vibration of the core']:  # Add other features if desired
            if feature_col in df.columns:
                new_col_mean = f'{feature_col}_roll_mean{window}'
                df[new_col_mean] = df[feature_col].rolling(window=window).mean().shift(1)
                engineered_features.append(new_col_mean)
            else:
                print(f"Warning: Column '{feature_col}' not found for rolling feature engineering.")

    # Difference features for Target
    df[f'{TARGET_COLUMN}_diff1'] = df[TARGET_COLUMN].diff(1)
    engineered_features.append(f'{TARGET_COLUMN}_diff1')

    print(f"Engineered features created: {engineered_features}")

    # e. Handle missing values (check again after feature engineering)
    # Lags and rolling windows will create NaNs at the beginning
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows due to NaNs from feature engineering.")
    print(f"Shape after dropping NaNs: {df.shape}")

    if df.empty:
        print("ERROR: DataFrame is empty after dropping NaNs. Check lag/window sizes or original data.")
        return None

    return df, BASE_FEATURES + engineered_features


# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    df_raw = load_data(FILE_PATH)

    if df_raw is not None:
        processed_output = process_data(df_raw.copy())  # Use a copy to keep raw df if needed

        if processed_output is not None:
            df_processed, all_feature_names = processed_output

            if not df_processed.empty:
                # Define features (X) and target (y)
                X = df_processed[all_feature_names]
                y = df_processed[TARGET_COLUMN]

                print(f"\nShape of X (features): {X.shape}")
                print(f"Shape of y (target): {y.shape}")
                print(f"\nFeatures being used for model: {X.columns.tolist()}")

                # --- Chronological Train-Test Split ---
                # For time series, we must split data chronologically
                test_size_ratio = 0.2  # e.g., last 20% of data for testing
                split_index = int(len(df_processed) * (1 - test_size_ratio))

                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                # Keep corresponding dates/CSN for plotting if needed
                dates_test = df_processed[DATE_COLUMN].iloc[split_index:]
                csn_test = df_processed[CSN_COLUMN].iloc[split_index:]

                print(f"\nTraining set size: {len(X_train)}")
                print(f"Test set size: {len(X_test)}")

                if len(X_train) == 0 or len(X_test) == 0:
                    print("ERROR: Training or test set is empty. Adjust test_size_ratio or check data.")
                else:
                    # --- Model Training (XGBoost) ---
                    print("\n--- Training XGBoost Regressor ---")
                    # Note: For best results, hyperparameter tuning is recommended (e.g., using GridSearchCV)
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',  # Common for regression
                        n_estimators=100,  # Number of trees, can be tuned
                        learning_rate=0.1,  # Can be tuned
                        max_depth=5,  # Can be tuned
                        subsample=0.8,  # Can be tuned
                        colsample_bytree=0.8,  # Can be tuned
                        random_state=42,  # For reproducibility
                        # early_stopping_rounds=10    # Optional: to prevent overfitting if you have a validation set
                    )

                    # If using early stopping, you'd need a validation set from the training data
                    # For simplicity, we'll train on the whole X_train for now
                    model.fit(X_train, y_train)
                    print("Model training complete.")

                    # --- Prediction ---
                    print("\n--- Making Predictions on Test Set ---")
                    y_pred = model.predict(X_test)

                    # --- Evaluation ---
                    print("\n--- Model Evaluation ---")
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    print(f"Mean Absolute Error (MAE): {mae:.4f}")
                    print(f"Mean Squared Error (MSE): {mse:.4f}")
                    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                    print(f"R-squared (R2 ): {r2:.4f}")

                    # --- Plotting Predictions vs Actual ---
                    print("\n--- Plotting Results ---")
                    plt.figure(figsize=(15, 7))
                    plt.plot(dates_test.values, y_test.values, label='Actual EGT Margin', marker='.', linestyle='-')
                    plt.plot(dates_test.values, y_pred, label='Predicted EGT Margin', marker='.', linestyle='--')

                    # Or plot against CSN if dates are too dense or not as relevant for x-axis
                    # plt.plot(csn_test.values, y_test.values, label='Actual EGT Margin', marker='.', linestyle='-')
                    # plt.plot(csn_test.values, y_pred, label='Predicted EGT Margin', marker='.', linestyle='--')

                    plt.title('EGT Margin Prediction vs Actual (Test Set)')
                    plt.xlabel('Flight DateTime (or CSN)')
                    plt.ylabel('EGT Margin')
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    # To save the plot:
                    # plt.savefig('egt_prediction_plot.png')
                    plt.show()

                    print("\n--- Script Finished ---")