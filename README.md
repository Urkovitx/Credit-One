Predictive Analysis of Credit Card Default
This repository contains a complete analysis and a Machine Learning model to predict the probability of default for credit card clients, based on the dataset from the Kaggle competition "Default of Credit Card Clients".

1. Project Objective
The main objective is to develop a robust classification model that proactively identifies clients with a high risk of failing to meet their payment obligations in the next month. The analysis focuses not only on the model's accuracy but also on its practical applicability in a business environment, proposing strategic actions based on the results.

2. Dataset
The dataset used comes from the Kaggle competition and contains demographic and financial behavior information for 30,000 clients of a bank in Taiwan. Key variables include:

Demographic Data: Sex, education level, marital status, and age.

Financial Data: Credit limit (LIMIT_BAL).

Payment History (PAY_): Payment status for the last 6 months.

Bill Amount (BILL_AMT_): Outstanding debt in the last 6 months.

Amount Paid (PAY_AMT_): Amount paid in the last 6 months.

Target Variable (default payment next month): 1 if the client defaults, 0 otherwise.

3. Methodology and Workflow
The project follows a structured workflow to ensure the quality and relevance of the results:

Exploratory Data Analysis (EDA): Analysis of variable distributions, identification of outliers, and study of correlations. A strong imbalance was detected in the target variable (only 22% of default cases).

Feature Engineering: Creation of new variables to capture more complex behavioral patterns, such as TOTAL_MONTHS_DELAYED and IS_CONSISTENT_PAYER. These new features have proven to be highly predictive.

Class Imbalance Management: Due to the low percentage of default cases, the SMOTE (Synthetic Minority Over-sampling Technique) has been applied to rebalance the training dataset. This allows the model to learn the patterns of the minority class more effectively.

Modeling and Optimization:

A Random Forest Classifier model was selected for its robustness and ability to handle non-linear relationships.

Hyperparameter optimization was performed using GridSearchCV with stratified cross-validation.

The chosen optimization metric was the F1-Score, ideal for problems with imbalanced classes.

Strategic Threshold Tuning: This has been one of the key points of the analysis. Instead of using the default decision threshold (0.5), an optimal threshold (0.3824) was calculated to maximize the F1-Score. This technique aligns the model with the business objective of maximizing the detection of at-risk clients (Recall).

4. Key Results
The final model demonstrates good predictive capability and, more importantly, great practical utility:

Overall Performance: The model achieves an AUC of 0.78, indicating a good ability to discriminate between the two classes.

Impact of the Optimized Threshold: Adjusting the decision threshold has increased the Recall for the default class (1) from 51% to 63%. This represents a relative improvement of 23.5% in our ability to detect clients who will actually default.

5. Business Recommendations
Based on the results, a Strategic Action Plan is proposed to manage at-risk clients, segmenting them according to the default probability predicted by the model and applying personalized interventions for each risk level (moderate, high, and very high).

Furthermore, it is recommended to enrich the model in future iterations with client income data to further improve its predictive power.

6. How to Run the Project
Requirements: Make sure you have Python 3 and the following libraries installed:

pip install pandas numpy sqlalchemy scikit-learn imbalanced-learn matplotlib seaborn jupyter

Database: The project uses a SQLite database named credit_one_dades.sqlite. This file must be in the same folder as the notebook to load the data.

Execution: Open and run the RFbestHP.ipynb notebook in a Jupyter environment to replicate the entire analysis and modeling process.

7. Technologies Used
Language: Python 3

Main Libraries:

Pandas and NumPy for data manipulation.

Scikit-learn for modeling and evaluation.

Imbalanced-learn for the SMOTE technique.

Matplotlib and Seaborn for data visualization.

SQLAlchemy for database connection.
