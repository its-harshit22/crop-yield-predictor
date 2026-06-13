# 🌾 Crop Yield Predictor

An interactive machine learning web app that predicts crop yield (hg/ha) based on country, crop type, year, rainfall, pesticide usage, and average temperature.

## 🔗 Live Demo
[https://crop-yield-predictore.streamlit.app/](https://crop-yield-predictore.streamlit.app/)

## 📋 Overview
This project uses historical agricultural data (1990–2013) across 100+ countries and 10 major crops to train a regression model that predicts crop yield based on environmental and agricultural inputs. It's deployed as a Streamlit web app where users can select a country, crop, and conditions to get an instant yield prediction.

## 🛠️ Tech Stack
- **Python**
- **pandas** – data loading and preprocessing
- **scikit-learn** – RandomForestRegressor model
- **Streamlit** – interactive web interface
- **matplotlib** – visualizations
- **joblib** – model serialization
- Deployed on **Streamlit Community Cloud**

## 📊 Dataset
FAO/World Bank crop yield dataset (Kaggle), containing:
- Country/Area
- Crop type (Item)
- Year
- Average rainfall (mm/year)
- Pesticide usage (tonnes)
- Average temperature (°C)
- Yield (hg/ha)

## 🤖 Model
- **Algorithm:** RandomForestRegressor
- **Preprocessing:** One-hot encoding for categorical features (Area, Crop)
- **Evaluation Metrics:**
  - R² Score: ~0.95+
  - MAE: ~3461 hg/ha

## 🚀 Features
- Select country, crop, year, rainfall, pesticide usage, and temperature
- Get real-time crop yield predictions
- View feature importance and model performance visualizations

## 📁 Project Structure
crop-yield-predictor/

├── app.py                  # Streamlit app

├── train_model.py          # Model training script

├── model.pkl               # Trained model

├── columns.pkl             # Feature columns for inference

├── yield_df.csv            # Dataset

├── actual_vs_predicted.png # Model performance visualization

├── feature_importance.png  # Feature importance chart

├── requirements.txt        # Dependencies

└── README.md
## 🔮 Future Work
- Integrate real-time weather/satellite data
- Extend into agricultural verification systems (e.g., comparing predicted vs. reported yield for crop damage compensation claims)
- Experiment with additional models (XGBoost, gradient boosting)

## 👤 Author
**Harshit Kushwah**
- [LinkedIn](https://www.linkedin.com/in/harshit-kushwah)
- [GitHub](https://github.com/its-harshit22)
