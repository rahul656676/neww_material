import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("insurance_predictions.csv")

# Show column names
print("Columns in the dataset:", df.columns)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Ensure necessary columns exist
required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'charges']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# Fill missing values
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])

# Convert to numeric
df['sex'] = df['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].astype(str).str.lower().map({'no': 0, 'yes': 1})

# Drop predicted column if it exists (retraining)
if 'predicted_charges' in df.columns:
    df = df.drop('predicted_charges', axis=1)

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train model
model = RandomForestRegressor(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-validation R² scores:", cv_scores)
print("Mean R² score:", np.mean(cv_scores))

# Hyperparameter tuning
params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X, y)

best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best R² score from GridSearchCV:", grid.best_score_)

# Predict and save
df['predicted_charges'] = best_model.predict(X)
df.to_csv("insurance_predictions.csv", index=False)
print("Saved predictions to insurance_predictions.csv")

# Save the model
joblib.dump(best_model, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")
