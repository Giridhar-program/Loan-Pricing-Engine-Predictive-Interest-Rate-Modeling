# Loan-Pricing-Engine-Predictive-Interest-Rate-Modeling

📌 Project Overview
A machine learning regression engine developed to automate the assignment of loan interest rates. This project demonstrates an end-to-end Software Development Life Cycle (SDLC) for a financial AI product—from synthetic data generation and feature engineering to model serialization and inference.

🛠️ Technical Stack
    
    Language: Python
    
    Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Joblib
    
    Model: Linear Regression (Standardized)
    
    Deployment: Serialized Pickle files (.pkl) for real-time inference

🏗️ Architecture & Features

    The model predicts a fair interest rate based on the following borrower attributes:
    
    Credit Score: Historical creditworthiness (300-850).
    
    Annual Income: Total yearly earnings.
    
    Loan Amount: Total principal requested.
    
    Loan-to-Income Ratio: A custom-engineered feature to measure borrower leverage.

📈 Key Engineering Steps

    Data Simulation: Generated a 1,000-sample dataset using NumPy with Gaussian noise to simulate real-world financial volatility.
    
    Feature Engineering: Engineered a loan_to_income_ratio to capture non-linear risk relationships.
    
    Standardization: Applied StandardScaler to ensure features with different units (e.g., $200,000 income vs. 800 credit score) were weighted fairly by the model.
    
    Guardrail Implementation: Developed a post-processing layer using np.clip to enforce business logic (interest rate floors and ceilings).

📊 Performance Metrics

    R² Score: 0.92+ (Explains over 92% of the variance in interest rates).
    
    Mean Squared Error (MSE): Low error rate indicating high precision against ground-truth formulas.


🚀 How to Run
Clone the repository:

    git clone https://github.com/your-username/loan-pricing-engine.git

Install dependencies:

    pip install numpy pandas scikit-learn joblib

Run the Interactive Inference Tool:

    python finance_model.py
