import streamlit as st
import pandas as pd
import joblib # o import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score # Añadir accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

try:
    model, labelEncoder = joblib.load("XentimentAnalyzer.pkl")
except FileNotFoundError:
    st.error("Error: The model file 'XentimentAnalyzer.pkl' was not found. Make sure the file exists in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

TRAIN_PATH = "data/twitter_training.csv"
TEST_PATH = "data/twitter_validation.csv"
SCRAP_DIR = "scrapfly/"

def loadStandardCSV(path):
    try:
        return pd.read_csv(path, names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"])
    except FileNotFoundError:
        st.error(f"Error: Data file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV file {path}: {e}")
        return None

def loadScrapflyCSV(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found: {path}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV file {path}: {e}")
        return None

def processData(df, le):
    requiredCols = ["Tweet_Content", "Sentiment"]
    if not all(col in df.columns for col in requiredCols):
        missingCols = [col for col in requiredCols if col not in df.columns]
        st.error(f"The CSV file must contain the columns: {', '.join(missingCols)}.")
        return None

    df = df.dropna(subset=["Tweet_Content", "Sentiment"]).copy()

    unknownSentiments = set(df['Sentiment']) - set(le.classes_)
    if unknownSentiments:
        st.warning(f"WARNING: Unknown sentiments were found in the data and will be ignored: {unknownSentiments}")
        df = df[df['Sentiment'].isin(le.classes_)]
        if df.empty:
            st.error("No valid data remains after filtering out unknown sentiments.")
            return None

    if not df['Tweet_Content'].isnull().all():
        df["Prediction_Num"] = model.predict(df["Tweet_Content"])
        df["Prediction"] = le.inverse_transform(df["Prediction_Num"])
        if not df.empty:
             df["Sentiment_Num"] = le.transform(df["Sentiment"])
        else:
             df["Sentiment_Num"] = pd.Series(dtype=int)

    else:
        st.warning("WARNING: The 'Tweet_Content' column contains only null values ​​after cleaning.")
        df["Prediction_Num"] = pd.Series(dtype=int)
        df["Prediction"] = pd.Series(dtype=str)
        df["Sentiment_Num"] = pd.Series(dtype=int)

    return df

st.set_page_config(page_title="XentimentAnalyzer Dashboard", layout="wide")
st.title("XentimentAnalyzer Dashboard")
st.markdown("Visualization and analysis of the performance of the trained sentiment model.")

source = st.sidebar.radio("Select data source:", ["Train", "Test", "Scrapfly"])

df_loaded = None

if source == "Train":
    df_loaded = loadStandardCSV(TRAIN_PATH)
elif source == "Test":
    df_loaded = loadStandardCSV(TEST_PATH)
else:
    if not os.path.exists(SCRAP_DIR):
         st.sidebar.error(f"Directory '{SCRAP_DIR}' does not exist.")
         st.stop()
    scrap_files = [f for f in os.listdir(SCRAP_DIR) if f.endswith(".csv")]
    if not scrap_files:
        st.sidebar.warning(f"WARNING: No .csv files were found in the directory '{SCRAP_DIR}'.")
        st.stop()
    else:
        selected = st.sidebar.selectbox("Choose a CSV file:", scrap_files)
        if selected:
            df_loaded = loadScrapflyCSV(os.path.join(SCRAP_DIR, selected))


if df_loaded is not None:
    df_processed = processData(df_loaded, labelEncoder)

    if df_processed is not None and not df_processed.empty:
        st.success(f"Data successfully loaded and processed ({len(df_processed)} filas).")

        if source in ["Train", "Test"]:
            st.subheader("Performance Metrics (vs Ground Truth)")
            if df_processed['Sentiment_Num'].nunique() >= 2 and df_processed['Prediction_Num'].nunique() >= 1:
                try:
                    col1, col2, col3 = st.columns(3)
                    report = classification_report(df_processed["Sentiment_Num"], df_processed["Prediction_Num"], target_names=labelEncoder.classes_, output_dict=True, zero_division=0)
                    conf_matrix = confusion_matrix(df_processed["Sentiment_Num"], df_processed["Prediction_Num"])

                    col1.metric("Accuracy", f"{report.get('accuracy', 0):.2%}")

                    total_fn = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
                    total_fp = np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)

                    col2.metric("False Negatives (Total)", int(np.sum(total_fn)))
                    col3.metric("False Positives (Total)", int(np.sum(total_fp)))

                    st.subheader("🔍 Matriz de Confusión")
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labelEncoder.classes_)
                    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
                    st.pyplot(fig)

                except ValueError as e_metrics:
                    st.error(f"Error calculating detailed metrics: {e_metrics}")
                    try:
                        accuracy = accuracy_score(df_processed["Sentiment_Num"], df_processed["Prediction_Num"])
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    except Exception:
                         st.metric("Accuracy", "Error")
                    st.info("Detailed metrics and confusion matrix cannot be displayed.")
                except Exception as e_general:
                    st.error(f"An unexpected error occurred while generating metrics: {e_general}")
                    st.info("Metrics and confusion matrix cannot be displayed.")

            else:
                st.warning("There are not enough distinct classes in the actual or predicted data to calculate detailed metrics.")
                try:
                    accuracy = accuracy_score(df_processed["Sentiment_Num"], df_processed["Prediction_Num"])
                    st.metric("Accuracy", f"{accuracy:.2%}")
                except Exception:
                     st.metric("Accuracy", "N/A")


        elif source == "Scrapfly":
            st.info("ℹScrapfly data contains the sentiment predicted during scraping. Performance metrics (Accuracy, FN, FP, Confusion Matrix) are not calculated for this source, as it does not represent a ground truth.")

        st.subheader("Sentiment Distribution")
        col4, col5 = st.columns(2)
        with col4:
            title_real = "Distribution (Predicted on CSV)" if source == 'Scrapfly' else "Distribution Real"
            st.markdown(f"**{title_real}**")
            st.bar_chart(df_processed["Sentiment"].value_counts())
        with col5:
            st.markdown("**Predicted distribution (Dashboard)**")
            st.bar_chart(df_processed["Prediction"].value_counts())

        st.subheader("Watch the post")
        if not isinstance(df_processed.index, pd.RangeIndex):
             df_processed = df_processed.reset_index(drop=True)

        if not df_processed.empty:
            selected_idx = st.selectbox("Select a post (index):", df_processed.index)
            if selected_idx in df_processed.index:
                 tweet = df_processed.loc[selected_idx]
                 st.markdown(f"**Original Text:** {tweet.get('Tweet_Content', 'N/A')}")
                 if 'Entity' in tweet:
                     st.markdown(f"**Entity:** `{tweet.get('Entity', 'N/A')}`") # Usar .get también aquí

                 sentiment_label = "Sentiment (Predicho en CSV)" if source == "Scrapfly" else "Sentimiento Real"
                 st.markdown(f"**{sentiment_label}:** `{tweet.get('Sentiment', 'N/A')}`")
                 st.markdown(f"**Predicted sentiment (Dashboard):** `{tweet.get('Prediction', 'N/A')}`")
            else:
                st.warning("Invalid selected index.")
        else:
             st.warning("There are no processed tweets to display.")

    elif df_processed is None:
         st.stop()
    else:
         st.warning("WARNING: The selected file does not contain valid data after processing.")


elif df_loaded is None and source == "Scrapfly":
     st.info("Select a CSV file from the 'scrapfly/' folder in the sidebar.")
elif df_loaded is None:
     st.info("Waiting for data to load...")


st.sidebar.info("Remember to run this application using: streamlit run XentimentDashboard.py")