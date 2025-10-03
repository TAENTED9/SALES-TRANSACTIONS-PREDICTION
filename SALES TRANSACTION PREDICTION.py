import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


try:
    # Download latest version
    path = kagglehub.dataset_download("srinivasav22/sales-transactions-dataset")
    print(f"\nDataset downloaded and saved in: {path}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# Step 2: Read the first 200 rows of the Excel file using pandas
df = pd.read_excel(path + r"\Test.xlsx")
#df = pd.read_excel(r"\.cache\kagglehub\datasets\srinivasav22\sales-transactions-dataset\versions\1\Test.xlsx")

# Save the Excel data to a CSV file
df.to_csv(r"~\Downloads\Downloaded_file.csv", index=False)

# Read only the first 200 rows
df_head = df.head(200)

# Step 3: Save these rows into another file called "For_Prediction.csv"
df_head.to_csv("For_Prediction.csv", index=False)

# Optionally, read back the new CSV file to verify
df = pd.read_csv("For_Prediction.csv")

df.to_csv(r"~\Downloads\For_Prediction.csv", index=False)

rows = df.shape[0]
cols = df.shape[1]
classes = df.iloc[:, -1].nunique()

print(f"\nFirst {rows} rows, {cols} columns and {classes} classes have been saved to 'For_Prediction.csv'.")

#----------------------------------------------------------------------------------

# Load the dataset
df = pd.read_csv(r'~\Downloads\For_Prediction.csv')

# Selecting features (X) and target variable (y)
X = df[['Quantity']]
y = df['TotalSalesValue']


# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train a linear regression model
model = LinearRegression().fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nThe Mean Absolute Error is: {mae}\nThe R² Score is: {r2}")


# ---------------------- PLOT 1: Actual vs Predicted Scatter ----------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.6)
plt.plot(X_test, y_pred, color='green', label='Regression Line')
plt.title("Actual vs Predicted Sales")
plt.xlabel("Quantity")
plt.ylabel("Total Sales Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot_actual_vs_predicted.png")
plt.show()


# ---------------------- PLOT 2: Residuals Plot ----------------------
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='gray', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Total Sales Value")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png")
plt.show()



