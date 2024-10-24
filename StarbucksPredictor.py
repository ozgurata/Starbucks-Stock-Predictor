'''
*******************************
Author:
u3289717 Assessment 3_Program_(a) 21/10/2024
*******************************
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import necessary libraries for machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Import joblib for saving the trained model
import joblib


# Title of the app
st.title("Starbucks Daily Stock Information")

# File uploader at the top
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Check if the uploaded file is a CSV
    if uploaded_file.name.endswith('.csv'):
        # Read the dataset
        data = pd.read_csv(uploaded_file)

        # Display dataset shape before removing duplicates
        original_shape = data.shape
        st.write("Dataset before deleting duplicate data:")
        st.write(f"Number of rows: {original_shape[0]}, Number of columns: {original_shape[1]}")

        # Remove duplicates
        data_cleaned = data.drop_duplicates()

        # Display dataset shape after removing duplicates
        cleaned_shape = data_cleaned.shape
        st.write("Dataset after deleting duplicate data:")
        st.write(f"Number of rows: {cleaned_shape[0]}, Number of columns: {cleaned_shape[1]}")

        # Show a sample of the cleaned dataset
        st.write("### Sample Data:")
        st.dataframe(data_cleaned.sample(10), use_container_width=True)

        # Explanatory text about the dataset
        st.write("### Key observations from Step 1 about Data Description")
        st.write("This file contains daily stock information for Starbucks.")
        st.write("There are 6 attributes and they are outlined below:")
        st.write("- **datetime**: The date of the stock information (formatted as YYYY-MM-DD).")
        st.write("- **open**: The stock price at market opening on that date.")
        st.write("- **high**: The highest stock price reached during the trading day.")
        st.write("- **low**: The lowest stock price reached during the trading day.")
        st.write("- **close**: The stock price at market closing on that date.")
        st.write("- **volume**: The number of shares traded on that date.")

        # Step 2: Problem Statement Definition
        st.write("### Step 2: Problem Statement Definition")
        st.write("Creating a prediction model to predict the closing price of Starbucks stock.")
        st.write("- **Target Variable:** `close`")
        st.write(
            "- **Predictors/Features:** `open`, `high`, `low`, `volume`, `datetime` (after converting to appropriate numerical features).")

        # Step 3: Choosing the Appropriate ML/AI Algorithm for Data Analysis
        st.write("### Step 3: Choosing the Appropriate ML/AI Algorithm for Data Analysis")
        st.write("Based on the problem statement, we need to create a supervised ML regression model, as the target variable (`close`) is continuous.")

        # Step 4: Analyzing the Target Variable Distribution
        st.write("### Step 4: Analyzing the Target Variable Distribution")
        st.write("Looking at the target variable's distribution (closing price) helps us check if the data is balanced or skewed.")
        st.write("If the target variable's distribution is too skewed, the predictive modeling may lead to poor results. Ideally, a bell curve is desirable, but slight positive or negative skew is acceptable.")

        # Creating histogram for the target variable 'close'
        plt.figure(figsize=(10, 5))
        plt.hist(data_cleaned['close'], bins=50, edgecolor='k', alpha=0.7)
        plt.title('Distribution of Closing Prices for Starbucks Stock')
        plt.xlabel('Closing Price ($)')
        plt.ylabel('Frequency')
        plt.grid(axis='x', alpha=0.75)
        plt.grid(axis='y', alpha=0.75)

        # Displaying the histogram in Streamlit
        st.pyplot(plt)

        # Observations from Step 4
        st.write("### Observations from Step 4: Analyzing the Target Variable Distribution")
        st.write("The distribution of the target variable, Starbucks' closing stock price, shows a slight positive skew.")
        st.write("However, this level of skewness should not significantly affect the performance of the machine learning algorithms we plan to use.")
        st.write("Additionally, the dataset appears to have a sufficient number of rows for each range of closing prices, ensuring that the model will have enough data to learn from and generate meaningful predictions.")

        # Step 5: Basic Exploratory Data Analysis (EDA)
        st.write("### Step 5: Basic Exploratory Data Analysis (EDA)")
        st.write("This step aims to gauge the overall structure of the Starbucks dataset, examine the types of columns present, and identify which ones are relevant to the target variable (`close`).")
        st.write("By looking at the data's overall shape and summary statistics, we can begin rejecting irrelevant columns that don't impact the target variable.")

        st.write("The key goals of this step are:")
        st.write("- Understanding the **volume of data** (number of rows and columns).")
        st.write("- Identifying which columns are **Quantitative**, **Categorical**, or **Qualitative**.")
        st.write("- Starting the process of **column rejection** by assessing if each column affects the `close` price of the stock. If a column has no impact, it will be removed; otherwise, it will be kept for further analysis.")

        # Display first few rows of the cleaned dataset
        st.write("### Sample Rows from the Data")
        st.dataframe(data_cleaned.head(), use_container_width=True)

        # Display summarized information about the data
        st.write("### Data Summary")
        buffer = io.StringIO()
        data_cleaned.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

        # Display descriptive statistics of the data
        st.write("### Descriptive Statistics")
        st.dataframe(data_cleaned.describe())

        # Identify if columns are categorical or continuous
        st.write("### Unique Value Counts for Each Column")
        st.dataframe(data_cleaned.nunique())

        # Observations from Step 5 – Basic Exploratory Data Analysis
        st.write("### Observations from Step 5 – Basic Exploratory Data Analysis")
        st.write("Based on this initial exploration, we can now create a simple report of the data and begin forming a roadmap for further analysis:")
        st.write("- **datetime**: Categorical (But does not fit for the scope of the project. Explained in Step 9.) Selected.")
        st.write("- **open**: Continuous. Selected.")
        st.write("- **high**: Continuous. Selected.")
        st.write("- **low**: Continuous. Selected.")
        st.write("- **close**: Continuous. This is the **Target Variable**, which will be predicted by our regression model. Selected.")
        st.write("- **volume**: Continuous. Selected.")
        st.write("In this step, the selected columns are not final; further analysis will refine the feature set for the model.")

        # Step 5.1: Removing Unwanted Columns
        #st.write("### Step 5.1: Removing Unwanted Columns")

        #Reasoning for Dropping the 'datetime' column
        #st.write("**Reasoning for Dropping the `datetime` Column:**")
        #st.write("The `datetime` column is neither a traditional categorical nor a quantitative variable. While it represents time points, it does not directly contribute to the prediction of the target variable (`close`) without being transformed into additional features like 'day of the week' or 'month.'")

        # Step 5.2: Visual Exploratory Data Analysis (EDA)
        st.write("### Step 5.2: Visual Exploratory Data Analysis (EDA)")
        st.write(
            "This step involves visualizing the distribution of all continuous predictor variables in the data using histograms.")
        st.write(
            "Visualizing these distributions helps understand the spread and nature of the data, such as whether the data is skewed or normally distributed.")

        # List of continuous variables to visualize
        continuous_columns = ['open', 'high', 'low', 'volume']

        # Loop through each continuous column to create histograms
        for column in continuous_columns:
            st.write(f"### Distribution of `{column}`")
            # Create histogram for each column
            plt.figure(figsize=(10, 5))
            plt.hist(data_cleaned[column], bins=50, edgecolor='k', alpha=0.7)
            plt.title(f'Distribution of {column.capitalize()}')
            plt.xlabel(f'{column.capitalize()}')
            plt.ylabel('Frequency')
            plt.grid(axis='x', alpha=0.75)
            plt.grid(axis='y', alpha=0.75)
            st.pyplot(plt)

        # Write observations for each histogram
        st.write("### Observations Summary from Step 5.2")
        st.write("- Histograms for the continuous predictor variables (`open`, `high`, `low`, and `volume`) provide insights into the spread of stock prices and trading volumes.")
        st.write("- Understanding these distributions is crucial before proceeding to model training as it helps identify data ranges and potential outliers that may affect predictions.")
        st.write("")
        st.write(f"- The distribution of `open`,`high` and `low` shows the general spread of values observed in the dataset.")
        st.write(f"- The `open`,`high` and `low` values show bell curves with slight positive skew which should not affect the ML algorithm.")
        st.write(f"- However, `volume` is strongly positively skewed which means we need to deal with outliers of this column for it to be eligible.")

        # Step 6: Outlier Analysis
        st.write("## Step 6: Outlier Analysis")
        st.write("We are going to find the outliers of each column using the IQR method.")
        st.write("If there are any outliers founded by that method, we are going to replace those outlier values with the upper bound value of their respected column.")

        # Select only continuous numerical columns
        numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])

        if not numeric_columns.empty:
            # Proceed with the outlier analysis if there are numeric columns
            Q1 = numeric_columns.quantile(0.25)
            Q3 = numeric_columns.quantile(0.75)
            IQR = Q3 - Q1

            # Identifying outliers using the IQR method
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (numeric_columns < lower_bound) | (numeric_columns > upper_bound)

            # Count outliers in each column
            outlier_count = outliers.sum()

            # Display the number of outliers in each column
            st.write("### Number of Outliers in Each Numeric Column")
            dtype_df_outlier = pd.DataFrame(outlier_count, columns=["Number of Outliers"]).reset_index()
            dtype_df_outlier = dtype_df_outlier.rename(columns={"index": "Column Name"})
            st.dataframe(dtype_df_outlier, use_container_width=True)

            # Handling outliers in the 'volume' column
            if 'volume' in numeric_columns.columns:
                st.write("## Handling Outliers in the `volume` Column")

                # Find the upper bound for non-outlier values in the volume column
                upper_bound_volume = int(upper_bound['volume'])

                st.write(f"The upper bound for `volume` is: {upper_bound_volume:,}")

                # Replace outliers in the volume column with the upper bound
                data_cleaned['volume'] = data_cleaned['volume'].apply(
                    lambda x: upper_bound_volume if x > upper_bound_volume else x
                )

                st.write("Outliers in the `volume` column have been replaced with the upper bound.")
                st.write(f"### Distribution of `volume` after replacing the outliers")
                # Create histogram for each column
                plt.figure(figsize=(10, 5))
                plt.hist(data_cleaned['volume'], bins=50, edgecolor='k', alpha=0.7)
                plt.title('Distribution of Volume (After outliers are treated)')
                plt.xlabel('Volume')
                plt.ylabel('Frequency')
                plt.grid(axis='x', alpha=0.75)
                plt.grid(axis='y', alpha=0.75)
                st.pyplot(plt)


            st.write("### Observations from Step 6: Outlier Analysis")
            st.write("- As expected, the `volume` column has outliers, with a total of **337 outlier values** identified.")
            st.write("- By removing the outliers, we reduce the building of bias in our machine learning models")
            st.write("- But in this case, because the number of outliers for `volume` is so high, it will still affect the correlation which we will investigate deeper to see if its a viable column.")
            st.write("- The other columns (`open`, `high`, `low`, `close`) do not have any identified outliers, suggesting that the stock price itself fluctuates within a more consistent range.")
        else:
            st.write("No continuous numeric columns available for outlier analysis.")

        # Step 7: Missing Value Analysis
        st.write("## Step 7: Missing Value Analysis")
        missing_values = data_cleaned.isnull().sum()
        st.write("### Missing Values in Each Column")
        dtype_df_missing_values = pd.DataFrame(missing_values, columns=["Missing Values"]).reset_index()
        dtype_df_missing_values = dtype_df_missing_values.rename(columns={"index": "Column Name"})
        st.dataframe(dtype_df_missing_values, use_container_width=True)

        st.write("### Step 7 Observations")
        st.write("No missing values in this data!")
        st.write("So theres no need to do any treatment to any data samples(rows).")

        # Step 8: Feature Selection - Visual and Statistical Correlation Analysis
        st.write("## Step 8: Feature Selection - Visual and Statistical Correlation Analysis")
        st.write(
            "This step involves visualizing the relationships between the continuous predictor variables and the target variable (`close`).")
        st.write(
            "We will use scatter plots to observe the trends and Pearson's correlation values to quantify the strength of these relationships.")

        # List of continuous columns (predictors) to analyze with the target variable 'close'
        continuous_cols = ['open', 'high', 'low', 'volume']

        # Loop through each continuous predictor variable to create scatter plots
        for predictor in continuous_cols:
            st.write(f"### Scatter Plot: `{predictor}` vs `close`")
            # Create scatter plot for each predictor against the target variable 'close'
            plt.figure(figsize=(10, 5))
            plt.scatter(data_cleaned[predictor], data_cleaned['close'], alpha=0.7, edgecolors='k')
            plt.title(f'{predictor.capitalize()} vs Close')
            plt.xlabel(f'{predictor.capitalize()}')
            plt.ylabel('Close')
            plt.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(plt)

            # Calculate and display Pearson's correlation coefficient
            correlation_value = data_cleaned[predictor].corr(data_cleaned['close'])
            st.write(f"**Pearson's Correlation between `{predictor}` and `close`**: {correlation_value}")

            # Write a brief interpretation of the correlation
            if correlation_value > 0.7:
                st.write("- This indicates a **strong positive** correlation.")
            elif correlation_value > 0.3:
                st.write("- This indicates a **moderate positive** correlation.")
            elif correlation_value > 0:
                st.write("- This indicates a **weak positive** correlation.")
            elif correlation_value < -0.7:
                st.write("- This indicates a **strong negative** correlation.")
            elif correlation_value < -0.3:
                st.write("- This indicates a **moderate negative** correlation.")
            else:
                st.write("- This indicates a **weak or no correlation**.")
            st.write("")

        # Summary of correlation analysis
        st.write("### Summary of Correlation Analysis")
        st.write("- Scatter plots provide a visual understanding of how each predictor relates to the `close` price.")
        st.write("- Pearson's correlation values help quantify the strength and direction of these relationships.")
        st.write("- Based on the analysis, we can identify `open`, `high` and `low` have stronger relationships with `close`.")
        st.write("This is because:")
        st.write("1. All three shows strong increasing trend with `close`.")
        st.write("2. Pearson Values for all three was extremely close to 1.00 which means perfect correlation.")
        st.write("So we can consider them for building our prediction model. On the other hand:")
        st.write("- Based on the analysis, we can identify `volume` to have very weak to no relationships with `close`.")
        st.write("This is because:")
        st.write("1. The scatter plot shows no trend.")
        st.write("2. Pearson Values was around 0.20 which means weak or no correlation.")
        st.write("So we can **not** consider it for building our prediction model.")

        # Step 9: Statistical Feature Selection using ANOVA for Categorical Variables
        st.write("## Step 9: Statistical Feature Selection (ANOVA for Categorical Variables)")
        st.write(
            "To determine if `datetime` should be used as a predictor, we will group it by `year_month` to simplify the analysis. This allows us to see if there are significant differences in `close` prices across different months without over-complicating the analysis with day-to-day fluctuations.")

        # Preparing the data for ANOVA test using `datetime` as a categorical variable
        data_cleaned['datetime'] = pd.to_datetime(data_cleaned['datetime'])
        data_cleaned['year_month'] = data_cleaned['datetime'].dt.to_period('M')  # Convert datetime to year-month periods

        # Visualize the relationship using a box plot with matplotlib
        st.write("### Box Plot: `year_month` vs `close`")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create box plot using matplotlib
        data_cleaned.boxplot(column='close', by='year_month', ax=ax, grid=False)
        ax.set_title('Year-Month vs Close')
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('Close Price')
        plt.suptitle('')  # Remove the automatic title to avoid overlap
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        st.pyplot(fig)

        # Perform ANOVA on the 'year_month' as a categorical variable and 'close' as the target
        st.write("### ANOVA Test: `year_month` vs `close`")
        anova_groups = data_cleaned.groupby('year_month')['close'].apply(list)
        f_val, p_val = f_oneway(*anova_groups)

        # Create a DataFrame for the ANOVA results
        anova_results = pd.DataFrame({
            "Categorical Variable": ["year_month"],
            "F-value": [f_val],
            "p-value": [p_val]
        })

        # Display the ANOVA results with high precision
        st.write("### ANOVA Results")
        st.write(
            "The following table shows the F-value and P-value for the ANOVA test between `year_month` and `close` with high precision.")
        st.dataframe(anova_results.style.format({"F-value": "{:.6f}", "p-value": "{:.6e}"}), use_container_width=True)

        # Interpret the ANOVA results
        st.write("### Interpretation of ANOVA Results")
        st.write(f"- **F-value:** {f_val:.6f}")
        st.write(f"- **p-value:** {p_val:.6e}")
        st.write("""
                **Explanation:**
                - The extremely high F-value suggests that there is substantial variation between the means of `close` prices across different `year_month` groups.
                - A p-value close to zero indicates that these variations are statistically significant. However, this does not necessarily mean that `year_month` is practically useful for predicting `close` prices.
                - In time-series data like stock prices, statistical significance often reflects the natural variability of prices over time rather than a stable relationship that can improve a predictive model.
                """)

        # Decision about using `datetime` as a predictor
        if p_val < 0.05:
            st.write(
                "**Conclusion:** `year_month` is statistically significantly correlated with `close` (p-value < 0.05). This suggests that changes in the `year_month` **might** have an effect on the `close` price.")
        else:
            st.write(
                "**Conclusion:** `year_month` is **NOT** statistically significantly correlated with `close` (p-value >= 0.05). This suggests that changes in `year_month` do not have a strong effect on the `close` price.")

        st.write("""
                **Decision to Exclude `datetime` as a Predictor:**
                - While the ANOVA test shows statistical significance, the relationship between `year_month` and `close` is not practically useful for a regression model.
                - Stock prices vary significantly over time due to external factors such as market conditions, economic changes, and company performance.
                - As a result, including `datetime` as a simple categorical predictor would not provide a stable or reliable input for predicting the stock's closing price.
                - Therefore, `datetime` is excluded from the model, focusing instead on numerical features like `open`, `high`, `low` for predicting `close`.
                """)
        st.write("In this project, our focus is on building a straightforward regression model using directly relevant numerical predictors.")
        st.write("By excluding the `datetime` column, we simplify the dataset, reduce potential noise, and maintain focus on features that have a more immediate impact on the stock's closing price. This approach ensures that the model remains interpretable and meets the project's objectives effectively.")
        st.write("Future studies could include time-based features for potentially improved prediction accuracy.")

        st.write("`datetime` column is removed.")

        # Dropping the 'datetime' column
        data_cleaned = data_cleaned.drop(columns=['datetime'])

        # Step 10: Selecting Final Predictors for Building Machine Learning Model
        st.write("## Step 10: Selecting Final Predictors for Building Machine Learning Model")
        st.write("Based on the analysis conducted in previous steps, we have identified the following features as the most relevant for predicting the `close` price:")
        st.write("- **`open`**: The stock price at market opening on a given day.")
        st.write("- **`high`**: The highest price the stock reached during the trading day.")
        st.write("- **`low`**: The lowest price the stock reached during the trading day.")
        st.write("These predictors were chosen based on their strong correlation with the `close` price and their ability to capture daily price movements in the stock market.")

        # Select the final dataset with only the chosen predictors and the target variable 'close'
        final_features = ['open', 'high', 'low']
        DataForML = data_cleaned[final_features + ['close']]
        st.write("### Final Data Subset for Model Training")
        st.dataframe(DataForML.head(), use_container_width=True)

        # Saving this final data subset for reference during deployment
        DataForML.to_pickle('DataForML.pkl')
        st.write("The final dataset has been saved for reference during the deployment phase as 'DataForML.pkl'. This ensures consistency between model training and deployment.")

        # Step 11: Data Conversion to Numeric Values for Machine Learning
        st.write("## Step 11: Data Conversion to Numeric Values for Machine Learning")
        st.write("In this step, we would convert any nominal (categorical) variables to numeric format, which is necessary for machine learning models to process the data.")

        # Explanation of categorical data handling (no actual conversion needed here)
        st.write("""
                **Explanation:**
                - Nominal variables are those which represent categories without any inherent order (e.g., colors, cities).
                - Ordinal variables are those which represent categories with a specific order (e.g., rating scales like 'Low', 'Medium', 'High').
                - In this dataset, the selected predictors `open`, `high`, and `low` are continuous numerical variables, and the target variable `close` is also continuous.
                - As such, there are **no nominal or ordinal variables** that require conversion to numeric values.
                """)

        # Conversion steps (if there were categorical variables)
        st.write("""
                **Note:**
                If we had categorical variables, the following steps would be performed:
                - Convert **ordinal categorical columns** to numeric using a mapping (e.g., `{'Low': 1, 'Medium': 2, 'High': 3}`).
                - Convert **binary nominal categorical columns** to numeric using 1/0 mapping.
                - Convert **other nominal categorical columns** to numeric using `pd.get_dummies()` to create binary indicator variables.
                """)

        # Summary before moving to data transformation
        st.write("### Observations for Step 11:")
        st.write("Since all our selected predictors are already numeric, no further conversion is needed. We can proceed directly to data transformation if required.")

        # Step 12: Train/Test Data Split and Standardization/Normalization of Data
        st.write("## Step 12: Train/Test Data Split and Standardization/Normalization of Data")
        st.write("""
        Splitting the data into training and testing samples is a crucial step in machine learning. 
        The training data is used to build the model, while the testing data is kept aside to evaluate the model's performance on unseen data. 
        This approach ensures that the model is not simply memorizing the training data but can generalize to new, unseen data.
        """)

        # Extracting the features and target variable
        X = DataForML[final_features]
        y = DataForML['close']

        # Splitting the data into training and testing sets
        st.write("""
        ### Splitting the Data:
        We use 70% of the data for training and the remaining 30% for testing.
        This split ensures that the model has enough data to learn from while retaining a portion to test its generalization.
        """)
        test_size = st.slider("Select the test size (percentage)", min_value=0.1, max_value=0.5, value=0.3,step=0.05)  # Default is 30%

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Displaying a sample of the training data
        st.write("### Sample Training Data:")
        st.dataframe(X_train.head(), use_container_width=True)

        # Step 12.1: Standardization/Normalization (Default Enabled)
        st.write("""
        ### Data Standardization/Normalization:
        - Standardization transforms the data to have a mean of 0 and a standard deviation of 1.
        - This is especially important for distance-based algorithms like **K-Nearest Neighbors (KNN)** and **Neural Networks**, which are sensitive to the scale of the data.
        - In this step, we use `StandardScaler` to standardize the data.
        """)

        # Explanation of `fit_transform` and `transform`
        st.write("""
        - **`fit_transform`**: This is used on the training data. It calculates the mean and standard deviation of the training data and then scales it accordingly.
        - **`transform`**: This is used on the testing data. It uses the mean and standard deviation calculated from the training data to scale the testing data.
        - This approach ensures that the testing data is scaled in the same way as the training data, preventing data leakage and ensuring fair evaluation of the model.
        """)

        # Providing a choice for the user to disable standardization, but it is enabled by default
        standardize_data = st.checkbox("Apply Standardization/Normalization", value=True)  # Default is checked (True)

        if standardize_data:
            st.write("Standardization has been applied to the training and testing data.")
            scaler = StandardScaler()
            # Use `fit_transform` on the training data
            X_train_scaled = scaler.fit_transform(X_train)
            # Use `transform` on the testing data
            X_test_scaled = scaler.transform(X_test)
            st.write("Data has been scaled using `StandardScaler` with mean=0 and standard deviation=1.")
            st.write("""
            **Range of Scaled Values:**
            - After standardization, the scaled values will have a mean of 0 and a standard deviation of 1.
            - Most of the values will lie between **-3** and **3**, but the range can vary depending on the original distribution of the data.
            - This transformation helps ensure that no single feature disproportionately influences the model.
            """)
        else:
            st.write("Standardization/Normalization has **not** been applied. Using the raw data for training.")
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Display a sample of the scaled training data
        st.write("### Sample of Scaled Training Data:")
        st.write("(Only shown if standardization is applied)")
        st.dataframe(X_train_scaled[:5] if standardize_data else X_train.head(), use_container_width=True)

        # Step 13: Model Training and Evaluation
        st.write("## Step 13: Model Training and Evaluation")
        st.write("""
                In this step, we will train multiple regression models using the training data and evaluate their performance using the test data.
                This allows us to compare different models and choose the one that provides the best predictions for the `close` price of Starbucks stock.
                """)

        # Define the regression models we want to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "SVM Regressor": SVR()
        }

        # Train the models and evaluate their performance
        model_performance = {}
        for name, model in models.items():
            st.write(f"### Training and Evaluating: {name}")
            # Fit the model on the training data
            model.fit(X_train_scaled, y_train)
            # Make predictions on the test data
            y_pred = model.predict(X_test_scaled)
            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            # Store the results in the dictionary
            model_performance[name] = {"MSE": mse, "R2 Score": r2, "MAE": mae}

        # Convert the performance dictionary to a pandas DataFrame for better visualization
        performance_df = pd.DataFrame(model_performance).T  # Transpose to get model names as rows

        # Apply styles using pandas built-in styling for better visualization
        styled_df = performance_df.style.format(precision=2) \
            .background_gradient(subset=["MSE"], cmap="Blues", low=0, high=1) \
            .background_gradient(subset=["R2 Score"], cmap="Greens", low=0, high=1) \
            .background_gradient(subset=["MAE"], cmap="Reds", low=0, high=1) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{
            'selector': 'th',
            'props': [('font-size', '14px'), ('text-align', 'center'), ('color', '#ffffff'),
                      ('background-color', '#404040')]
        }])

        # Display the table with model performance metrics
        st.write("## Model Performance Table")
        st.dataframe(styled_df, use_container_width=True)

        # Step 13.1: Visualizing the Performance Comparison between Models
        st.write("## Step 13.1: Visualizing Model Performance Comparison")
        st.write("""
                We will use bar charts to visualize the performance of each model based on three metrics:
                - **MSE** (Mean Squared Error): Lower values indicate better performance.
                - **R2 Score**: Values closer to 1 indicate better fit.
                - **MAE** (Mean Absolute Error): Lower values indicate better performance.
                """)

        # Extracting model names and their respective performance metrics
        model_names = list(model_performance.keys())
        mse_values = [model_performance[model]["MSE"] for model in model_names]
        r2_values = [model_performance[model]["R2 Score"] for model in model_names]
        mae_values = [model_performance[model]["MAE"] for model in model_names]

        # Creating bar plots to compare MSE, R2, and MAE across models
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        # MSE Comparison
        ax[0].bar(model_names, mse_values, color='blue')
        ax[0].set_title("Model Comparison: Mean Squared Error (MSE)")
        ax[0].set_ylabel("MSE")
        ax[0].tick_params(axis='x', rotation=45)

        # R2 Score Comparison
        ax[1].bar(model_names, r2_values, color='green')
        ax[1].set_title("Model Comparison: R2 Score")
        ax[1].set_ylabel("R2 Score")
        ax[1].tick_params(axis='x', rotation=45)

        # MAE Comparison
        ax[2].bar(model_names, mae_values, color='red')
        ax[2].set_title("Model Comparison: Mean Absolute Error (MAE)")
        ax[2].set_ylabel("MAE")
        ax[2].tick_params(axis='x', rotation=45)

        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)

        # Step 14: Selecting the Best Model
        st.write("## Step 14: Selecting the Best Model")
        st.write("""
                Based on the evaluation metrics from the previous step, we will select the model with the lowest Mean Squared Error (MSE).
                The model with the lowest MSE provides the best predictions on average, making it the best choice for our regression task.
                """)

        # Check if model performance dictionary has been populated
        if model_performance:
            # Select the model with the lowest MSE
            best_model_mse = min(model_performance, key=lambda x: model_performance[x]["MSE"])
            st.write(f"### Best Model based on Lowest Mean Squared Error (MSE): `{best_model_mse}`")

            # Step 14.2: Retraining the Best Model on the Entire Dataset
            st.write("## Step 14.2: Retraining the Best Model")
            st.write("""
                    Now that we have identified the best model, we will retrain it using **100% of the data** to maximize its predictive power.
                    This step ensures that the model learns from the entire dataset before deployment.
                    """)

            # Retrieve the best model instance
            best_model = models[best_model_mse]

            # Check if standardization was applied earlier
            if standardize_data:
                st.write("Standardization was applied, using scaled data for retraining.")
                # Combine and scale the entire dataset using the scaler fitted on the training data
                X_combined_scaled = scaler.fit_transform(X)
                best_model.fit(X_combined_scaled, y)
            else:
                st.write("Standardization was not applied, using raw data for retraining.")
                # Use raw data without scaling
                best_model.fit(X, y)

            st.write(f"The model `{best_model_mse}` has been retrained on the entire dataset.")

            # Step 14.3: Saving the Best Model
            st.write("## Step 14.3: Saving the Best Model")
            st.write("""
                    Saving the trained model allows us to deploy it and use it for making predictions on new data without needing to retrain.
                    The model is saved as a serialized file (`.pkl`), which can be loaded back into the application or shared with others.
                    """)

            # Save the best model as a serialized file using joblib
            model_filename = "best_model.pkl"
            joblib.dump(best_model, model_filename)
            st.write(f"Model `{best_model_mse}` has been saved as `{model_filename}`. You can use this file for deployment.")

        else:
            st.write("No model performance results available. Please ensure models were trained successfully.")

        # Step 15: Model Deployment - Load the Saved Model and Predict
        st.write("## Step 15: Model Deployment - Predict Using Saved Model")

        # Load the saved model
        model_filename = "best_model.pkl"
        try:
            # Load the model from the file
            loaded_model = joblib.load(model_filename)
            st.write(f"Model `{model_filename}` loaded successfully!")

            # Use a form to control the input process and prediction
            with st.form("prediction_form"):
                st.write("### Provide the input values for prediction")
                # Generate input fields dynamically based on the selected features
                user_input_values = {}
                for feature in final_features:  # Ensure `final_features` from Step 10 is available
                    user_input_values[feature] = st.number_input(
                        f"Enter value for {feature}",
                        value=float(DataForML[feature].mean())  # Default to the mean value of the feature
                    )

                # Add a submit button to make predictions
                submit_button = st.form_submit_button(label="Predict")

            # If the submit button is clicked, proceed with the prediction
            if submit_button:
                # Convert the user inputs into a DataFrame
                user_input_df = pd.DataFrame([user_input_values])

                # Check if standardization was applied earlier
                if standardize_data:
                    st.write("Standardization was applied earlier, scaling the input values before prediction.")
                    # Scale the user inputs using the previously used scaler
                    user_input_scaled = scaler.transform(user_input_df)
                else:
                    user_input_scaled = user_input_df

                # Make predictions using the loaded model
                predicted_value = loaded_model.predict(user_input_scaled)

                # Display the predicted value
                st.write(f"### Predicted close: {predicted_value[0]:.2f}")

        except FileNotFoundError:
            st.write(f"Model `{model_filename}` not found. Please ensure the model has been saved correctly.")

    else:
        st.error("Please upload a valid CSV file.")
