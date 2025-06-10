import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from script_de_prediction import load_data, process_data, predict_future_egt
from script_evaluation import load_data as load_data_eval, process_data as process_data_eval

# Configuration
st.set_page_config(page_title="EGT Margin Analysis", layout="wide")

# Titre de l'application
st.title("Analyse des Marges EGT")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ["Évaluation", "Prédiction"])

# --- Ajout UI : Chargement des données dans la sidebar ---
st.sidebar.markdown("#### Chargement des données")
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop file here",
    type=["csv"],
    accept_multiple_files=False,
    help="Limit 200MB per file - CSV"
)

if uploaded_file is not None:
    st.sidebar.success(f"Fichier chargé : {uploaded_file.name}")
else:
    st.sidebar.info("Aucun fichier chargé.")

# Fonction pour charger et traiter les données
@st.cache_data
def load_and_process_data(uploaded_file=None):
    if uploaded_file is None:
        return None, None
    
    try:
        df_raw = pd.read_csv(uploaded_file)
        if df_raw is not None:
            processed_output = process_data(df_raw.copy())
            if processed_output is not None:
                return processed_output
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
    return None, None

# Page de prédiction
if page == "Prédiction":
    st.header("Prédiction des Marges EGT")
    
    # Charger les données
    df_processed, all_feature_names = load_and_process_data(uploaded_file)
    
    if df_processed is not None and all_feature_names is not None:
        # Paramètres de prédiction
        st.subheader("Paramètres de prédiction")
        num_future_steps = st.slider("Nombre de cycles à prédire", 5, 50, 25)
        
        # Bouton de prédiction
        predict_btn = st.button("🚀 Lancer la prédiction", type="primary", disabled=uploaded_file is None)
        
        if predict_btn:
            # Entraîner le modèle
            X = df_processed[all_feature_names]
            y = df_processed['EGT Margin']
            
            test_size_ratio = 0.2
            split_index = int(len(df_processed) * (1 - test_size_ratio))
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Faire les prédictions
            last_actual_csn = df_processed['CSN'].iloc[-1]
            last_actual_csslv = df_processed['Cycles Since last shop visit'].iloc[-1]
            historical_sequence_data = df_processed.iloc[-10:]
            
            future_df = predict_future_egt(
                model, historical_sequence_data, num_future_steps,
                X_train.columns.tolist(),
                last_known_csn=last_actual_csn,
                last_known_csslv=last_actual_csslv
            )
            
            # Afficher les résultats
            st.subheader("Prédictions futures")
            
            # Graphique des prédictions avec zone critique
            fig = go.Figure()
            
            # Ajouter la zone critique (rouge)
            fig.add_trace(go.Scatter(
                x=future_df['CSN'],
                y=[12] * len(future_df),
                fill=None,
                mode='lines',
                line_color='rgba(255, 0, 0, 0)',
                name='Zone critique'
            ))
            fig.add_trace(go.Scatter(
                x=future_df['CSN'],
                y=[18] * len(future_df),
                fill='tonexty',
                mode='lines',
                line_color='rgba(255, 0, 0, 0.2)',
                name='Zone critique'
            ))
            
            # Ajouter les prédictions
            fig.add_trace(go.Scatter(
                x=future_df['CSN'],
                y=future_df['EGT Margin'],
                mode='lines+markers',
                name='Prédictions',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title='Prédictions des Marges EGT',
                xaxis_title='CSN',
                yaxis_title='EGT Margin',
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bouton de téléchargement des prédictions
            csv = future_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name="predictions_egt_margin.csv",
                mime="text/csv"
            )
            
            # Tableau des prédictions
            st.dataframe(future_df)
            
            # Métriques (facultatif)
            st.subheader("Métriques de performance")
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{mae:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("R²", f"{r2:.4f}")

# Page d'évaluation
elif page == "Évaluation":
    st.header("Évaluation du Modèle")
    
    # Charger les données
    df_processed, all_feature_names = load_and_process_data(uploaded_file)
    
    if df_processed is not None and all_feature_names is not None:
        # Paramètres d'évaluation
        # st.subheader("Paramètres d'évaluation")
        test_size = 40
        
        # Bouton d'évaluation
        eval_btn = st.button("📊 Lancer l'évaluation", type="primary", disabled=uploaded_file is None)
        
        if eval_btn:
            # Entraîner le modèle
            X = df_processed[all_feature_names]
            y = df_processed['EGT Margin']
            
            test_size_ratio = test_size / 100
            split_index = int(len(df_processed) * (1 - test_size_ratio))
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Faire les prédictions
            y_pred = model.predict(X_test)
            
            # Calculer les métriques
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Afficher les métriques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("MSE", f"{mse:.2f}")
            with col3:
                st.metric("RMSE", f"{rmse:.2f}")
            with col4:
                st.metric("R²", f"{r2:.2f}")
            
            # Afficher le graphique de comparaison
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=y_test,
                name='Valeurs réelles',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=y_pred,
                name='Prédictions',
                line=dict(color='red')
            ))
            fig.update_layout(
                title='Comparaison des valeurs réelles et prédites',
                xaxis_title='CSN',
                yaxis_title='EGT Margin',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Veuillez charger un fichier CSV pour commencer l'évaluation.") 