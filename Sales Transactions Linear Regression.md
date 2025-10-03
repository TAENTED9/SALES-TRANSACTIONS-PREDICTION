# Sales Transactions Prediction Dataset

This project downloads a sales transactions dataset from Kaggle, processes the data, and applies linear regression to predict total sales value based on quantity. It also generates plots for actual vs predicted values and residuals.

## Features

- Downloads dataset from Kaggle using `kagglehub`
- Reads and processes 9000+ Excel data with pandas
- Saves processed data to CSV files
- Trains a linear regression model using scikit-learn
- Evaluates model performance (MAE, R²)
- Generates scatter and residual plots

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## Usage

1. Install dependencies:

```bash
    pip install -r requirements.txt
```

2. Run the script:
    
```bash
    python "SALES TRANSACTION PREDICTION.py"
```

3. Ensure you have access to Kaggle datasets (API credentials may be required).

## Output

- `For_Prediction.csv`: Processed data for prediction
- `scatter_plot_actual_vs_predicted.png`: Scatter plot of actual vs predicted sales
- `residuals_plot.png`: Residuals plot
