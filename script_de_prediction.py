# 0. IMPORTS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

# --- Configuration ---
TARGET_COLUMN = 'EGT Margin'
DATE_COLUMN = 'Flight DateTime'
CSN_COLUMN = 'CSN'
BASE_FEATURES = ['Vibration of the core', CSN_COLUMN, 'Cycles Since last shop visit']
MAX_LOOKBACK = 10


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
    if df is None: return None
    print("\n--- Starting Data Processing ---")
    if DATE_COLUMN not in df.columns:
        print(f"ERROR: Date column '{DATE_COLUMN}' not found.")
        return None
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    print(f"'{DATE_COLUMN}' converted to datetime.")

    cols_to_convert_numeric = [TARGET_COLUMN] + BASE_FEATURES
    for col in cols_to_convert_numeric:
        if col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
            elif not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Column '{col}' is now numeric (dtype: {df[col].dtype}).")

    df = df.sort_values(by=[DATE_COLUMN, CSN_COLUMN]).reset_index(drop=True)
    print("DataFrame sorted by DATE_COLUMN and CSN_COLUMN.")

    print("Starting feature engineering...")
    engineered_features = []
    lags_target = [1, 2, 3, 5, 7];
    lags_features = [1, 2, 3]
    for lag in lags_target:
        new_col_name = f'{TARGET_COLUMN}_lag{lag}';
        df[new_col_name] = df[TARGET_COLUMN].shift(lag);
        engineered_features.append(new_col_name)
    for feature_col in ['Vibration of the core']:
        if feature_col in df.columns:
            for lag in lags_features:
                new_col_name = f'{feature_col}_lag{lag}';
                df[new_col_name] = df[feature_col].shift(lag);
                engineered_features.append(new_col_name)
    window_sizes = [5, 10]
    for window in window_sizes:
        new_col_mean = f'{TARGET_COLUMN}_roll_mean{window}';
        new_col_std = f'{TARGET_COLUMN}_roll_std{window}'
        df[new_col_mean] = df[TARGET_COLUMN].rolling(window=window).mean().shift(1)
        df[new_col_std] = df[TARGET_COLUMN].rolling(window=window).std().shift(1)
        engineered_features.extend([new_col_mean, new_col_std])
        for feature_col in ['Vibration of the core']:
            if feature_col in df.columns:
                new_col_mean_feat = f'{feature_col}_roll_mean{window}';
                df[new_col_mean_feat] = df[feature_col].rolling(window=window).mean().shift(1);
                engineered_features.append(new_col_mean_feat)
    df[f'{TARGET_COLUMN}_diff1'] = df[TARGET_COLUMN].diff(1);
    engineered_features.append(f'{TARGET_COLUMN}_diff1')
    print(f"Engineered features created: {len(engineered_features)} features")
    all_feature_names_for_model = BASE_FEATURES + engineered_features
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows due to NaNs from feature engineering.")
    print(f"Shape after dropping NaNs: {df.shape}")
    if df.empty: print("ERROR: DataFrame is empty after dropping NaNs."); return None, None
    return df, all_feature_names_for_model


# --- 4. FUNCTION TO PREDICT FUTURE CSN ---
def predict_future_egt(model, last_n_historical_data, num_future_steps, feature_names_ordered, last_known_csn,
                       last_known_csslv):
    future_predictions = []
    current_egt_sequence = last_n_historical_data[TARGET_COLUMN].tolist()
    current_vibration_sequence = last_n_historical_data['Vibration of the core'].tolist()
    last_actual_vibration = current_vibration_sequence[-1]
    current_csn = last_known_csn
    current_csslv = last_known_csslv
    print("\n--- Starting Future Predictions ---")
    for step in range(1, num_future_steps + 1):
        features_for_step = {}
        current_csn += 1;
        current_csslv += 1
        features_for_step[CSN_COLUMN] = current_csn
        features_for_step['Cycles Since last shop visit'] = current_csslv
        features_for_step['Vibration of the core'] = last_actual_vibration
        lags_target_config = [1, 2, 3, 5, 7]
        for lag in lags_target_config:
            features_for_step[f'{TARGET_COLUMN}_lag{lag}'] = current_egt_sequence[-lag] if len(
                current_egt_sequence) >= lag else np.nan
        lags_features_config = [1, 2, 3]
        for lag in lags_features_config:
            features_for_step[f'Vibration of the core_lag{lag}'] = current_vibration_sequence[-lag] if len(
                current_vibration_sequence) >= lag else np.nan
        window_sizes_config = [5, 10]
        temp_egt_series = pd.Series(current_egt_sequence)
        for window in window_sizes_config:
            features_for_step[f'{TARGET_COLUMN}_roll_mean{window}'] = temp_egt_series.iloc[-window:].mean() if len(
                temp_egt_series) >= window else np.nan
            features_for_step[f'{TARGET_COLUMN}_roll_std{window}'] = temp_egt_series.iloc[-window:].std() if len(
                temp_egt_series) >= window else np.nan
        temp_vibration_series = pd.Series(current_vibration_sequence)
        for window in window_sizes_config:
            features_for_step[f'Vibration of the core_roll_mean{window}'] = temp_vibration_series.iloc[
                                                                            -window:].mean() if len(
                temp_vibration_series) >= window else np.nan
        features_for_step[f'{TARGET_COLUMN}_diff1'] = current_egt_sequence[-1] - current_egt_sequence[-2] if len(
            current_egt_sequence) >= 2 else np.nan
        feature_values_list = [features_for_step.get(fname, np.nan) for fname in feature_names_ordered]
        X_future_step = pd.DataFrame([feature_values_list], columns=feature_names_ordered)
        predicted_egt = model.predict(X_future_step)[0]
        future_predictions.append({CSN_COLUMN: current_csn, TARGET_COLUMN: predicted_egt})
        print(f"  Step {step}: Predicted EGT for CSN {current_csn} = {predicted_egt:.4f}")
        current_egt_sequence.append(predicted_egt)
        if len(current_egt_sequence) > MAX_LOOKBACK: current_egt_sequence.pop(0)
        current_vibration_sequence.append(last_actual_vibration)
        if len(current_vibration_sequence) > MAX_LOOKBACK: current_vibration_sequence.pop(0)
    return pd.DataFrame(future_predictions)


# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    df_raw = load_data(FILE_PATH)
    if df_raw is not None:
        processed_output = process_data(df_raw.copy())
        if processed_output is not None:
            df_processed, all_feature_names_for_model = processed_output
            if df_processed is not None and not df_processed.empty:
                X = df_processed[all_feature_names_for_model]
                y = df_processed[TARGET_COLUMN]
                print(f"\nFeatures being used for model: {X.columns.tolist()}")

                test_size_ratio = 0.2
                split_index = int(len(df_processed) * (1 - test_size_ratio))
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                print(f"\nTraining set size: {len(X_train)}")
                print(f"Test set size: {len(X_test)}")

                if len(X_train) > 0 and len(X_test) > 0:
                    print("\n--- Training XGBoost Regressor ---")
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                        max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42,
                    )
                    model.fit(X_train, y_train)
                    print("Model training complete.")

                    print("\n--- Model Evaluation (on historical test set) ---")
                    y_pred_test = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred_test);
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test));
                    r2 = r2_score(y_test, y_pred_test)
                    print(f"Mean Absolute Error (MAE): {mae:.4f}");
                    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}");
                    print(f"R-squared (R2 ): {r2:.4f}")

                    last_actual_csn = df_processed[CSN_COLUMN].iloc[-1]
                    last_actual_csslv = df_processed['Cycles Since last shop visit'].iloc[-1]

                    historical_sequence_data = df_processed.iloc[-MAX_LOOKBACK:] if len(
                        df_processed) >= MAX_LOOKBACK else df_processed

                    future_df = predict_future_egt(
                        model, historical_sequence_data, num_future_steps=25,
                        feature_names_ordered=X_train.columns.tolist(),
                        last_known_csn=last_actual_csn, last_known_csslv=last_actual_csslv
                    )
                    print("\n--- Predicted Future EGT Margins ---");
                    print(future_df)

                    # --- MODIFIED PLOTTING SECTION ---
                    print("\n--- Plotting Future Forecast ONLY ---")
                    plt.figure(figsize=(12, 6))  # Adjusted figure size for a single focused plot

                    if future_df is not None and not future_df.empty:
                        plt.plot(future_df[CSN_COLUMN].values, future_df[TARGET_COLUMN].values,
                                 label='Future Predicted EGT Margin (Next 25 CSN)',
                                 marker='o',  # Changed marker for better visibility
                                 linestyle='-',  # Changed to solid line
                                 color='green')
                        plt.title('Future EGT Margin Forecast (Next 25 CSN)')
                        plt.xlabel('CSN (Cycles Since New)')
                        plt.ylabel('EGT Margin')
                        plt.legend()
                        plt.grid(True)
                    else:
                        plt.text(0.5, 0.5, 'No future predictions to plot.', horizontalalignment='center',
                                 verticalalignment='center')
                        plt.title('Future EGT Margin Forecast')

                    plt.tight_layout()
                    plt.show()

                    print("\n--- Script Finished ---")
                else:
                    print("ERROR: Training or test set is empty after split. Cannot proceed with training.")
            else:
                print("ERROR: Processed DataFrame is empty. Cannot proceed.")