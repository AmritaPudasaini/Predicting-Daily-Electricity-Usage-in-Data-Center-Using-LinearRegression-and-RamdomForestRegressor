# Predicting-Daily-Electricity-Usage-in-Data-Center-Using-LinearRegression-and-RamdomForestRegressor

# Data Center Electricity Consumption Prediction

Predicts daily electricity usage of global data centers using machine learning regression.

## Dataset
- **Source:** [Global Data Center & AI Water/Electricity Usage](https://www.kaggle.com/datasets/ashyou09/global-data-center-and-ai-waterelectricity-usage) (Kaggle, by `ashyou09`)
- 126,770 rows × 14 columns, 2019–2025, no missing values
- **Target:** `Daily_Electricity_Usage_MWh`

## Workflow
EDA → Model Training → Hyperparameter Tuning → Cross-Validation

## Key Findings
- Capacity (MW) is the strongest predictor of electricity usage (r ≈ 0.98)
- Hyperscale/AI facilities use 13× more electricity than Enterprise/Standard facilities
- Relationship between features and usage is non-linear

## Models & Results
| Model | R² Score |
|---|---|
| Linear Regression | 0.6151 |
| Ridge Regression | 0.6151 |
| **Random Forest** | **0.9940** |

## Best Parameters (via GridSearchCV, 5-fold CV)
- `n_estimators = 100`
- `max_depth = 15`
- CV R² = 0.9938 (closely matches test R², confirming good generalization)

## Conclusion
Random Forest is the best model. Facility scale (capacity) - not efficiency - is the main driver of electricity demand.

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook DataCenter_Electricity.ipynb
```
