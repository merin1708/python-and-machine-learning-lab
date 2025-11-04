import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# ---------------- Single Linear Regression ----------------
print("----- Single Linear Regression (TV vs Sales) -----")
advertising_df = pd.read_csv("exp7/advertising.csv")
X_single = advertising_df[['TV']]
y_single = advertising_df['Sales']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)

model_single = LinearRegression()
model_single.fit(X_train_s, y_train_s)
y_pred_s = model_single.predict(X_test_s)

print("R² Score (Single Linear):", r2_score(y_test_s, y_pred_s))
print("Mean Squared Error (Single Linear):", mean_squared_error(y_test_s, y_pred_s))

# Plot Single Linear Regression: test data and predicted line
plt.figure(figsize=(7, 4))
plt.scatter(X_test_s['TV'], y_test_s, color='blue', alpha=0.6, label='Actual')
# Create a line across the TV range
tv_range = np.linspace(X_single['TV'].min(), X_single['TV'].max(), 100)
pred_line = model_single.predict(pd.DataFrame({'TV': tv_range}))
plt.plot(tv_range, pred_line, color='red', linewidth=2, label='Predicted')
plt.title('Single Linear Regression: TV vs Sales')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)  # Brief pause to let the figure render

# --- Interactive Prediction (Single Linear) ---
print("\nEnter new data for prediction (Single Linear: TV vs Sales)")
try:
    tv_val = float(input("Enter TV Advertisement spend ($): "))
    new_tv = pd.DataFrame({'TV': [tv_val]})
    pred_sales = model_single.predict(new_tv)
    print(f"Predicted Sales: {pred_sales[0]:.2f}")
except Exception as e:
    print("Invalid input, skipping prediction.", e)


# ---------------- Multiple Linear Regression ----------------
print("\n----- Multiple Linear Regression (Boston Housing) -----")
boston_df = pd.read_csv("exp7/Boston.csv")
X_multi = boston_df.drop(columns=['medv'])
y_multi = boston_df['medv']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

print("R² Score (Multiple Linear):", r2_score(y_test_m, y_pred_m))
print("Mean Squared Error (Multiple Linear):", mean_squared_error(y_test_m, y_pred_m))

# Plot Multiple Linear Regression: actual vs predicted scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_test_m, y_pred_m, alpha=0.6)
min_val = min(y_test_m.min(), y_pred_m.min())
max_val = max(y_test_m.max(), y_pred_m.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.title('Multiple Linear Regression: Actual vs Predicted (MEDV)')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)  # Brief pause to let the figure render

# --- Interactive Prediction (Multiple Linear) ---
print("\nEnter new data for prediction (Multiple Linear: Boston Housing)")
try:
    # collect inputs for ALL features
    inputs = {}
    for col in X_multi.columns:
        val = float(input(f"Enter value for {col}: "))
        inputs[col] = [val]

    new_house = pd.DataFrame(inputs)
    pred_price = model_multi.predict(new_house)
    print(f"Predicted MEDV (House Price): {pred_price[0]:.2f}")
except Exception as e:
    print("Invalid input, skipping prediction.", e)


# ---------------- Polynomial Regression ----------------
print("\n----- Polynomial Regression (Month vs Sales) -----")
ice_df = pd.read_csv("exp7/icecream.csv")
X_poly = ice_df[['month']]
y_poly = ice_df['sales']

poly = PolynomialFeatures(degree=2)
X_poly_transformed = poly.fit_transform(X_poly)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly_transformed, y_poly, test_size=0.2, random_state=42
)

model_poly = LinearRegression()
model_poly.fit(X_train_p, y_train_p)
y_pred_p = model_poly.predict(X_test_p)

print("R² Score (Polynomial Regression):", r2_score(y_test_p, y_pred_p))
print("Mean Squared Error (Polynomial Regression):", mean_squared_error(y_test_p, y_pred_p))

# Plot Polynomial Regression fit
plt.figure(figsize=(7, 4))
months_sorted = np.linspace(X_poly['month'].min(), X_poly['month'].max(), 100)
months_trans = poly.transform(pd.DataFrame({'month': months_sorted}))
sales_fit = model_poly.predict(months_trans)
plt.scatter(X_poly['month'], y_poly, color='blue', alpha=0.5, label='Data')
plt.plot(months_sorted, sales_fit, color='red', linewidth=2, label='Polynomial Fit')
plt.title('Polynomial Regression: Month vs Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)  # Brief pause to let the figure render

# --- Interactive Prediction (Polynomial Regression) ---
print("\nEnter new data for prediction (Polynomial: Month vs Sales)")
try:
    month_val = float(input("Enter Month (1-12): "))
    new_month = pd.DataFrame({'month': [month_val]})
    new_month_trans = poly.transform(new_month)
    pred_sales = model_poly.predict(new_month_trans)
    print(f"Predicted Ice Cream Sales: {pred_sales[0]:.2f}")
except Exception as e:
    print("Invalid input, skipping prediction.", e)

# Keep all plots open at the end
plt.show(block=True)
