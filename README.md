# Retail Sales Prediction

An end-to-end regression project that predicts `total_sales` for retail store/product combinations using store, product, and pricing features. The workflow covers data cleaning, exploratory data analysis, feature engineering, multicollinearity checks, and model comparison between Linear Regression and XGBoost.

## 📁 Project Structure

```
.
├── Redo.ipynb          # Main analysis and modeling notebook
├── train.csv            # Training data (not included — see Data section)
├── test.csv             # Test data (not included — see Data section)
└── submit14.csv          # Generated predictions (output)
```

## 📊 Dataset

The notebook expects two CSV files, `train.csv` and `test.csv`, containing store and product-level records with fields such as:

- `product_code`, `product_category`, `product_weight_kg`, `product_price`
- `fat_content`, `shelf_visibility`
- `store_code`, `store_size`, `store_location_tier`, `store_format`, `store_age_years`
- `total_sales` (target variable, present only in `train.csv`)

## 🔧 Workflow

1. **Data Cleaning**
   - Checked for duplicates and missing values
   - Standardized inconsistent text in `product_category`
   - Imputed missing `product_weight_kg` with the column mean
   - Imputed missing `store_size` with the mode (`Medium`)

2. **Exploratory Data Analysis**
   - Sales trends by fat content, product category, store, and store format
   - Store-level product sales breakdown (faceted bar charts)
   - Sales distribution by store format (pie chart)
   - Correlation heatmap of numeric features

3. **Feature Engineering**
   - `visibility_mean_ratio`: product's shelf visibility relative to its category average
   - `price_per_kg`: price normalized by product weight
   - `price_vs_category_mean`: price relative to its category's average price

4. **Multicollinearity Check**
   - Computed Variance Inflation Factor (VIF) for numeric features
   - Dropped `price_vs_category_mean`, `price_per_kg`, and `visibility_mean_ratio` after they showed high VIF

5. **Encoding**
   - One-hot encoded categorical features with `OneHotEncoder`

6. **Modeling**
   - **Linear Regression** — baseline model
   - **XGBoost Regressor** — tuned with custom hyperparameters (`max_depth`, `learning_rate`, `subsample`, regularization terms, etc.)

7. **Prediction & Submission**
   - Generated predictions on the held-out test set
   - Exported results to `submit14.csv` with `id` and `total_sales` columns

## 📈 Results

| Model              | R² Score | RMSE      |
|--------------------|----------|-----------|
| Linear Regression  | 0.419    | 1,301.80  |
| XGBoost Regressor  | 0.625    | 1,045.36  |

XGBoost outperformed the linear baseline on both R² and RMSE.

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

## ▶️ Usage

1. Place `train.csv` and `test.csv` in a known directory.
2. Update the file paths in the notebook's data-loading cells.
3. Run the notebook top to bottom in Jupyter:

```bash
jupyter notebook Redo.ipynb
```

4. The final predictions will be saved to a CSV file (e.g. `submit14.csv`).
