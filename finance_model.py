
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler=StandardScaler()
np.random.seed(42)

n_samples= 1000

noise=np.random.normal(0,1,n_samples)
credit_scores=np.random.randint(300,851,n_samples)
income=np.random.randint(20000,200001,n_samples)
loan_amount=np.random.randint(1000,50001,n_samples)
rate=25-(credit_scores/50)-(income/25000)+noise

data_dict={
    'credit_scores':credit_scores,
    'annual_income':income
    }


data_dict['loan_amount']=loan_amount
data_dict['interest_rate']=rate

df=pd.DataFrame(data_dict)
df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
df['interest_rate']=df['interest_rate'].clip(lower=1.5)
print(df.head(),df.isnull().sum(),df.describe())

X=df[['credit_scores','annual_income','loan_amount', 'loan_to_income_ratio']]
y=df[['interest_rate']]

X_scaled=scaler.fit_transform(X)
print(X_scaled)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse,r2)
print(model.coef_,model.intercept_)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel('Actual Interest Rate')
plt.ylabel('Predicted Interest Rate')
plt.title('Actual vs. Predicted Interest Rate')
plt.show()

"""**Actual vs. Predicted Interest Rate Scatter Plot: This scatter plot specifically shows how well your linear regression model performed:**

The x-axis represents the Actual Interest Rate (y_test values).
The y-axis represents the Predicted Interest Rate (y_pred values).
Each point on the plot corresponds to an actual interest rate from your test set and the rate your model predicted for it.
The red diagonal line indicates where actual values perfectly match predicted values (i.e., y_test = y_pred).
The closer the scattered points fall to this red line, the more accurate your model's predictions are. If the points are widely dispersed from the line, it indicates a less accurate model. In this case, the points are clustered fairly close to the red line, which aligns with the high R-squared value I observed, suggesting a good fit.
"""

# Create raw data
customers_raw = np.array([
    [450, 30000, 40000, 40000/30000], # Customer A
    [800, 150000, 10000, 10000/150000] # Customer B
])

# Scale the data using the SAME scaler we used for training
customers_scaled = scaler.transform(customers_raw)

# Predict!
predictions = model.predict(customers_scaled)
print(f"Customer A Rate: {predictions[0][0]:.2f}%")
print(f"Customer B Rate: {predictions[1][0]:.2f}%")

import joblib

joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

import joblib
import numpy as np

# 1. Load the "frozen" model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

print("--- AI Loan Pricing Tool ---")

while True:
    print("\nEnter borrower details (or type 'exit' to quit):")

    try:
        score = input("Credit Score (300-850): ")
        if score.lower() == 'exit': break

        income = input("Annual Income ($): ")
        loan = input("Loan Amount ($): ")

        # 2. DATA PREPROCESSING
        # Convert inputs to floats and calculate the ratio we engineered earlier
        score, income, loan = float(score), float(income), float(loan)
        ratio = loan / income

        # 3. FORMAT FOR MODEL
        # The model expects a 2D array: [[score, income, loan, ratio]]
        raw_data = np.array([[score, income, loan, ratio]])

        # 4. SCALE THE DATA
        scaled_data = scaled_data = scaler.transform(raw_data)

        # 5. PREDICT
        prediction = model.predict(scaled_data)

        print(f"--- RECOMMENDED INTEREST RATE: {prediction[0][0]:.2f}% ---")

    except ValueError:
        print("Please enter valid numbers.")

print("Tool closed.")
