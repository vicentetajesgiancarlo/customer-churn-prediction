# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecommunications company using machine learning. It includes a comprehensive data analysis notebook and a ready-to-use web application for real-time predictions.

## Project Structure

- `notebooks/`: Contains the Jupyter notebooks for data exploration, preprocessing, and model training.
  - `churn_prediccion.ipynb`: Main analysis in Spanish.
- `app/`: A FastAPI web application to serve predictions.
  - `app.py`: Backend server logic.
  - `index.html`: Web interface.
  - `static/`: CSS and JavaScript files for the UI.
- `data/`: Dataset used for training (Telco Customer Churn).
- `models/`: Trained models and scalers saved in `.joblib` format.
- `requirements.txt`: List of dependencies required to run the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vicentetajesgiancarlo/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Analysis and Model Training
Explore the `notebooks/churn_prediccion.ipynb` to see the full data science pipeline, from EDA to model evaluation (XGBoost and Logistic Regression).

### 2. Web Application
To start the prediction web app, run:
```bash
python app/app.py
```
Then open your browser at `http://localhost:8010`. The interface allows you to input customer data and receive an instant churn risk assessment.

## Technologies Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation.
- **Scikit-Learn & XGBoost**: Machine Learning models.
- **Matplotlib & Seaborn**: Data visualization.
- **FastAPI**: Backend web framework.
- **Vanilla HTML/CSS/JS**: Frontend interface.

## Dataset
The project uses the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
