import streamlit as st
import pandas as pd 
import mlflow.sklearn 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


st.set_page_config(
    page_title="Viesnīcas rezervāciju prognozēšana",
    layout="centered"
)

# --- Funkcija datu sagatavošanai ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    st.info("Notiek datu kopas sagatavošana")
    
    
   # Jā vai nē kolonnas
    yes_no_cols = [
            col for col in df.columns
            if df[col].astype(str).str.lower().isin(["yes", "no"]).any()
    ]

    # Mērvienību tirīšana
    def safe_to_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str).str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0],
                errors="coerce",
        )

    # Atlasīt numeriskas kolonnas
    cols_with_numbers = [
        col
        for col in df.columns
        if df[col].astype(str).str.contains(r"\d").any() and col not in yes_no_cols
    ]

    for col in cols_with_numbers:
        df[col] = safe_to_numeric(df[col])

    #Apmācībai apstrāde
    target_col = "booking_status"  
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    return df
   

# --- MLflow Model Loading with Caching ---
@st.cache_resource
def load_mlflow_model():
    try:
        # Full model URI format: models:/<model_name>/<version_alias>
        MODEL_NAME = "rf_champion_hotel"
        MODEL_ALIAS = "champion_hotel"
        full_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        st.info(f"Ielādējam modeli: **{full_uri}**")
        
        model = mlflow.sklearn.load_model(full_uri)
        st.success("Modelis tika veiksmīgi ielādēts.")
        return model
    except Exception as e:
        st.error(f"Error loading MLflow model: {e}")
        st.warning("Ensure your MLFLOW_TRACKING_URI is correct and the model 'rf_champion' with alias 'champion' exists.")
        # Return the placeholder model on failure so the app can still demonstrate the flow
        return model


# --- Main Streamlit App Logic ---
def main():
    st.title("Viesnīcas rezervāciju prognozēšana")
    
    st.subheader("1. Augšupielādējiet datni ar datiem")
    st.markdown("Augšupielādējiet datni ar datiem, kur sākuma rinda satur kolonnu sarakstu.")
    
    # Augšupielāde
    uploaded_file = st.file_uploader(
        "Izvēlietise CSV datni",
        type="csv"
    )

    if uploaded_file is not None:
        try:
            # 2. Load data into DataFrame
            st.subheader("2. Datu ielāde")
            
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data_df = pd.read_csv(stringio)
            
            st.dataframe(data_df.head())
            st.success(f"Veiksmīgi ielādēti {len(data_df)} ieraksti.")
        
            st.subheader("Statusu sadalījums")
            img = mpimg.imread('artifacts/data_booking_status_distribution.png')
            st.image(img, use_container_width=True)
            
            # 3. Preprocess the Data
            st.subheader("3. Datu sagatavošana")
            preprocessed_df = preprocess_data(data_df)
            
            st.dataframe(preprocessed_df.head())
            st.success("Datu sagatavošana pabeigta.")
            
            # 4. Load the Model
            st.subheader("4. Rezervācijas statusu prognozēšana")
            
            # Load the model (cached)
            model = load_mlflow_model()
            
            if model is not None:
                st.info("Prognozējam vērtības...")
                
                # 5. Predict target feature values
                predictions = model.predict(preprocessed_df)
                
                # 6. Prepare and Display Results
                results_df = pd.DataFrame({
                    "Record Index": data_df.booking_status,
                    "Forecasted Value": predictions
                })
                
                st.subheader("5. Rezultāti")
                st.dataframe(results_df)
                st.success("Prognozēšana pabeigta un rezultāti parādīti!")
        
            st.markdown("**Darba procesā tika secināts, ka datu kopai “Viesnīcas” vispiemērotākais modelis ir “Random Forest”.**")
            
            st.subheader("6. Testu datu kopas pārpratumu matrica")
            img = mpimg.imread('artifacts/CM_RandomForest.png')
            st.image(img, use_container_width=True) 
               
            st.subheader("7.1. Modeļu salīdzināšana pēc 'Accuracy'")
            img = mpimg.imread('artifacts/model_comparison_acc.png')
            st.image(img, use_container_width=True) 
            
            st.subheader("7.2. Modeļu salīdzināšana pēc 'AUC Score'")
            img = mpimg.imread('artifacts/model_comparison_auc.png')
            st.image(img, use_container_width=True) 

            st.subheader("7.3. Modeļu salīdzināšana pēc 'F1 Score'")
            img = mpimg.imread('artifacts/model_comparison_f1.png')
            st.image(img, use_container_width=True)        
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Please ensure your CSV file is correctly formatted and contains the expected features for the ML model.")

if __name__ == "__main__":
    main()
