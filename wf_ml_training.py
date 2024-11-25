import pandas as pd
import numpy as np
import os

# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)

# Function 1: Classify import dependency by category
def classify_import_dependency(imports, exports, mod):
    import joblib
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np

    # Aggregate dollar values by state, fiscal year, and fiscal quarter
    imports_grouped = imports.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    exports_grouped = exports.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    
    # Merge import and export data
    combined = pd.concat([imports_grouped, exports_grouped], axis=1, keys=["Imports", "Exports"]).fillna(0)

    # Reset index to include "State", "Fiscal year", and "Fiscal quarter" as columns
    combined = combined.reset_index()

    # Calculate import-to-export ratio
    combined["Ratio"] = combined["Imports"] / combined["Exports"]
    combined["Ratio"] = combined["Ratio"].replace([np.inf, -np.inf], 0)
    
    # Label dependency level as high, medium, or low
    bins = [0, 0.75, 2.0, float('inf')]
    labels = ['Low', 'Medium', 'High']
    combined['Dependency Level'] = pd.cut(combined['Ratio'], bins=bins, labels=labels, include_lowest=True)
    
    # Encode the dependency level as numeric
    le = LabelEncoder()
    combined["Encoded Label"] = le.fit_transform(combined["Dependency Level"])

    # Define features and target
    categorical_features = ["State", "Fiscal year", "Fiscal quarter"]
    numerical_features = ["Ratio"]
    X = combined[categorical_features + numerical_features]
    y = combined["Encoded Label"]

        # Convert X (features) to DataFrame before saving
    X_df = pd.DataFrame(X)  # Convert numpy array to DataFrame
    X_df.to_csv("models/X_train_import_dependency.csv", index=False)

    # Convert y (target) to Series before saving
    y_series = pd.Series(y)  # Convert numpy array to Series
    y_series.to_csv("models/y_train_import_dependency.csv", index=False)

    # Preprocessing: One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),  # One-hot encode "State", "Fiscal year", "Fiscal quarter"
        ],
        remainder="passthrough"  # Keep numerical features as is
    )

    # Define pipeline with preprocessing and model
    if mod == 'dependency1':
        model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=1,        # Minimal number of trees for simplicity
            max_depth=2,           # Very shallow depth for reduced complexity
            min_samples_split=2,   # Default splitting criterion
            min_samples_leaf=2,    # Larger leaf size to generalize better
            random_state=42
        ))
    ])

    elif mod == 'dependency2':
        model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=3,        # Slightly more trees for robustness
            max_depth=3,           # Shallow depth for simpler patterns
            min_samples_split=3,   # Slightly larger split to avoid overfitting
            min_samples_leaf=2,    # Keep leaf size consistent
            random_state=42
        ))
    ])

    elif mod == 'dependency3':
        model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=5,       # Moderate number of trees for ensemble
            max_depth=4,           # Restrict depth for simplicity
            min_samples_split=4,   # Larger splits for reduced complexity
            min_samples_leaf=3,    # Leaf size tuned for generalization
            random_state=42
        ))
    ])

    elif mod == 'dependency4':
        model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=10,       # Fewer trees to keep computation light
            max_depth=6,           # Allow slightly deeper splits for patterns
            min_samples_split=3,   # Moderate split size for generalization
            min_samples_leaf=5,    # Small leaves for finer resolution
            random_state=42
        ))
    ])

    # Train the model
    model.fit(X, y)

    if mod == 'dependency1':
        filename = "models/import_dependency_dependency1.joblib"
    elif mod == 'dependency2':
        filename = f"models/import_dependency_dependency2.joblib"
    elif mod == 'dependency3':
        filename = f"models/import_dependency_dependency3.joblib"
    elif mod == 'dependency4':
        filename = f"models/import_dependency_dependency4.joblib"
    else:
        raise ValueError("Unknwn Model")

    with open(filename, "wb") as f:
        joblib.dump(model, f)

    label_encoder = "models/import_dependency_label_encoder.joblib"
    with open(label_encoder, "wb") as f:
        joblib.dump(le, f)

# Function 2: Analyze seasonal fluctuations by category
def analyze_seasonal_fluctuations(data, trade_type, mod):
    import joblib
    import pandas as pd
    from sklearn.neural_network import MLPRegressor
    
    def get_season(quarter):
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        return seasons.get(quarter, 'Unknown')
    
    # Add 'Season' and 'Year' to the dataset
    data["Season"] = data["Fiscal quarter"].apply(get_season)
    
    # Aggregate by state, category, year, quarter, and season
    seasonal_data = data.groupby(["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"])["Dollar value"].sum().reset_index()
    
    # Create dummies for categorical features
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"]], drop_first=True)
    
    seasonal_data_values = seasonal_data["Dollar value"].values  # Extract the values column separately
    
    # Features (X) and target (y)
    X = seasonal_data_dummies
    y = seasonal_data_values

    # Convert X (features) to DataFrame before saving
    X_df = pd.DataFrame(X)  # Convert numpy array to DataFrame
    X_df.to_csv(f"models/X_train_seasonal_{trade_type.lower()}.csv", index=False)

    # Convert y (target) to Series before saving
    y_series = pd.Series(y)  # Convert numpy array to Series
    y_series.to_csv(f"models/y_train_seasonal_{trade_type.lower()}.csv", index=False)
    
    # Define and train the XGBoost model
    if mod == 'seasonal1':
        model = MLPRegressor(
        hidden_layer_sizes=(50,),   # Single hidden layer with 50 neurons
        activation='relu',          # ReLU activation function
        solver='adam',              # Adam optimizer for training
        max_iter=1000,              # Maximum iterations for training
        random_state=42
    )

    elif mod == 'seasonal2':
        model = MLPRegressor(
    hidden_layer_sizes=(64, 32),     # Two hidden layers, with 64 neurons in the first and 32 in the second
    activation='tanh',               # Tanh activation function for non-linearity
    solver='adam',                   # Adam optimizer (good default choice for neural networks)
    max_iter=1000,                   # Maximum number of iterations for training
    learning_rate='constant',        # Constant learning rate
    learning_rate_init=0.001,        # Initial learning rate (used for 'constant' learning rate)
    random_state=42                  # For reproducibility
)
        
    elif mod == 'seasonal3':
        model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),  # Three hidden layers with different sizes
        activation='tanh',                  # Tanh activation for smoother gradients
        solver='adam',                      # Adam optimizer for training
        max_iter=1000,                      # Maximum iterations
        early_stopping=True,                # Stop training when validation score doesn't improve
        random_state=42
    )

    elif mod == 'seasonal4':
        model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),   # Three layers with decreasing number of neurons
        activation='relu',                  # ReLU activation for better performance
        solver='adam',                      # Adam optimizer
        max_iter=1000,                      # Maximum iterations
        random_state=42
    )
        
    model.fit(X, y)

    # Save the model and the feature column names used during training
    
    if mod == 'seasonal1':
        filename = f"models/seasonal_{trade_type.lower()}_seasonal1.joblib"
    elif mod == 'seasonal2':
        filename = f"models/seasonal_{trade_type.lower()}_seasonal2.joblib"
    elif mod == 'seasonal3':
        filename = f"models/seasonal_{trade_type.lower()}_seasonal3.joblib"
    elif mod == 'seasonal4':
        filename = f"models/seasonal_{trade_type.lower()}_seasonal4.joblib"

    with open(filename, "wb") as f:
        joblib.dump(model, f, compress=9)
    
    # Save the column names used for training (for future reference)
    columns_filename = f"models/seasonal_columns_{trade_type.lower()}.joblib"
    with open(columns_filename, "wb") as f:
        joblib.dump(seasonal_data_dummies.columns, f)

# Main entry for testing
if __name__ == "__main__":
    # Load training data
    imports_train = pd.read_csv("data_processed/import_train.csv")
    exports_train = pd.read_csv("data_processed/export_train.csv")

