import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load and preprocess data
import_train = pd.read_csv('data_processed/import_train.csv')
export_train = pd.read_csv('data_processed/export_train.csv')

# Group by State and Fiscal year to calculate total import and export values
imports_sum = import_train.groupby(['State', 'Fiscal year', 'Fiscal quarter'])['Dollar value'].sum().reset_index()
exports_sum = export_train.groupby(['State', 'Fiscal year', 'Fiscal quarter'])['Dollar value'].sum().reset_index()

# Merge import and export data
merged_data = pd.merge(imports_sum, exports_sum, on=['State', 'Fiscal year', 'Fiscal quarter'], suffixes=('_import', '_export'))

# Calculate import-export ratio
merged_data['import_export_ratio'] = merged_data['Dollar value_import'] / merged_data['Dollar value_export'].replace(0, 1)

# Classify states into dependency categories
merged_data['dependency_category'] = pd.qcut(merged_data['import_export_ratio'], q=3, labels=['Low', 'Medium', 'High'])

# Add seasonal features based on the fiscal quarter
def get_season(quarter):
    if quarter in [1]:
        return 'Winter'
    elif quarter in [2]:
        return 'Spring'
    elif quarter in [3]:
        return 'Summer'
    else:
        return 'Fall'

merged_data['season'] = merged_data['Fiscal quarter'].map(get_season)

# Encode categorical variables
le_state = LabelEncoder()
le_season = LabelEncoder()
merged_data['State_encoded'] = le_state.fit_transform(merged_data['State'])
merged_data['season_encoded'] = le_season.fit_transform(merged_data['season'])

# Prepare features and target
X = merged_data[['Dollar value_import', 'Dollar value_export', 'import_export_ratio', 'season_encoded', 'State_encoded']]
y = merged_data['dependency_category']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_scaled, y)

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the model, scaler, and label encoders
joblib.dump(rf_classifier, 'models/agricultural_import_dependency_classifier.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_state, 'models/label_encoder_state.pkl')
joblib.dump(le_season, 'models/label_encoder_season.pkl')

print("Training complete")
