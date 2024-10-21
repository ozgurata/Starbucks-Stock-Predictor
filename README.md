# Starbucks Stock Price Predictor

Welcome to the **Starbucks Stock Price Predictor** project! This repository contains a machine learning model designed to predict the closing price of Starbucks stock based on historical stock data. It is built using Python and the **Streamlit** library, providing an interactive user interface for analyzing data, training models, and making predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features and Steps](#features-and-steps)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Model Deployment](#model-deployment)
- [License](#license)

## Project Overview
The goal of this project is to predict the **closing stock price** of Starbucks using various regression models. The dataset includes daily stock information with features such as `open`, `high`, `low`, `volume`, and `datetime`. By analyzing this data, we build and evaluate multiple regression models and select the best one for deployment.

### Key Objectives:
- Data preprocessing and exploratory data analysis (EDA).
- Identification and treatment of outliers.
- Feature selection and model training.
- Evaluation of multiple regression models.
- Interactive user interface for predictions using **Streamlit**.

## Project Structure
- **data/**: Folder for storing the stock data CSV file.
- **StarbucksPredictor.py**: Main Streamlit app script.
- **best_model.pkl**: Serialized best model for deployment.
- **README.md**: This README file.
- **LICENSE**: Project license.

## Features and Steps
This project follows a step-by-step approach to build, evaluate, and deploy a predictive model:

1. **Data Cleaning**: Remove duplicates and handle missing values.
2. **Exploratory Data Analysis**: Visualize the target variable and perform basic EDA.
3. **Outlier Analysis**: Identify and treat outliers in the data.
4. **Feature Selection**: Analyze relationships between features and the target variable.
5. **Missing Value Analysis**: Analyze and treat if there are any missing values.
6. **Model Training**: Train multiple regression models such as Linear Regression, Decision Tree, Random Forest, KNN, and SVM.
7. **Model Evaluation**: Evaluate models using metrics like Mean Squared Error (MSE), R² score, and Mean Absolute Error (MAE).
8. **Model Deployment**: Provide a user interface for inputting feature values and generating predictions with the best model.

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/ozgurata/Starbucks-Stock-Predictor.git
   cd Starbucks-Stock-Predictor

2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run StarbucksPredictor.py

4. Upload the stock data CSV file when prompted in the app interface.


## Usage
After starting the Streamlit app, you can:
- Upload a dataset containing daily Starbucks stock information.
- Visualize data distributions and perform feature selection.
- Train multiple regression models and evaluate their performance.
- Use the best model to predict the closing stock price based on user inputs.

**Example:**
1. Input values for open, high and low.
2. Click Predict to get the estimated closing stock price.
3. View visualizations of input features alongside the predicted output.

## Technologies Used
- Python: Core programming language.
- Streamlit: For building the interactive web interface.
- pandas: Data manipulation and analysis.
- scikit-learn: Machine learning models and evaluation metrics.
- Matplotlib: Data visualization.

## Model Deployment
The best-performing model is saved as best_model.pkl and is used for making predictions through the Streamlit interface. The saved model can be reloaded at any time to make predictions without retraining.
### Deployment Options:
- Local Deployment: Use **Streamlit** locally by following the installation steps.
- Cloud Deployment: Host the **Streamlit** app on platforms like **Heroku, AWS, or Streamlit Cloud** for easy access.

## License
This project is licensed under the GNU License. See the LICENSE file for details.

