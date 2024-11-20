import pandas as pd
import joblib

# Load the test data
import_test = pd.read_csv('data_processed/import_test.csv')
export_test = pd.read_csv('data_processed/export_test.csv')

# Group by State and Fiscal year to calculate total import and export values
imports_sum = import_test.groupby(['State', 'Fiscal year', 'Fiscal quarter'])['Dollar value'].sum().reset_index()
exports_sum = export_test.groupby(['State', 'Fiscal year', 'Fiscal quarter'])['Dollar value'].sum().reset_index()

# Merge import and export data
data = pd.merge(imports_sum, exports_sum, on=['State', 'Fiscal year', 'Fiscal quarter'], suffixes=('_import', '_export'))

# Calculate import-export ratio
data['import_export_ratio'] = data['Dollar value_import'] / data['Dollar value_export'].replace(0, 1)

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

data['season'] = data['Fiscal quarter'].map(get_season)

# Load label encoders and encode State and season
le_state = joblib.load('models/label_encoder_state.pkl')
le_season = joblib.load('models/label_encoder_season.pkl')
data['State_encoded'] = le_state.transform(data['State'])
data['season_encoded'] = le_season.transform(data['season'])

# Prepare the input features
features = data[['Dollar value_import', 'Dollar value_export', 'import_export_ratio', 'season_encoded', 'State_encoded']]

# Load the saved model and scaler
model = joblib.load('models/agricultural_import_dependency_classifier.pkl')
scaler = joblib.load('models/scaler.pkl')

# Scale the features
features_scaled = scaler.transform(features)

# Make predictions
predictions = model.predict(features_scaled)

# Add predictions to the original dataframe
data['predicted_category'] = predictions

# Display the results
print(data[['State', 'Fiscal year', 'Fiscal quarter', 'season', 'predicted_category']])

print("Predictions complete")
