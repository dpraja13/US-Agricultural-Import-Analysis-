import pandas as pd
import numpy as np
import os
#from wf_ml_evaluation import dependency1, dependency2, dependency3, dependency4, seasonal1, seasonal2, seasonal3, seasonal4

# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)

# Function 1: Classify import dependency by category
def classify_import_dependency(imports, exports, mod = None):
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
    if mod == None:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
    n_estimators=100,        # Default number of trees
    max_depth=10,            # Moderate depth to avoid overfitting
    min_samples_split=2,     # Default splitting criterion
    min_samples_leaf=1,      # Default leaf size
    random_state=42
))
        ])

    elif mod == 'dependency2':
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
    n_estimators=200,        # More trees for better ensemble learning
    max_depth=20,            # Allow deeper splits for more complex patterns
    min_samples_split=5,     # Increase minimum split samples for broader splits
    min_samples_leaf=3,      # Require more samples per leaf
    random_state=42
))
        ])

    elif mod == 'dependency3':
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
    n_estimators=150,        # Moderate number of trees
    max_depth=5,             # Restrict depth for simplicity
    min_samples_split=10,    # Require larger splits to reduce overfitting
    min_samples_leaf=5,      # Larger leaf size to generalize better
    random_state=42
))
        ])

    elif mod == 'dependency4':
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
    n_estimators=300,        # High number of trees for stability
    max_depth=None,          # No depth limit to capture complex patterns
    min_samples_split=2,     # Default split size to allow detailed splits
    min_samples_leaf=1,      # Allow small leaves for capturing fine details
    random_state=42
))
        ])

    # Train the model
    model.fit(X, y)

    '''if mod is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif isinstance(mod, RandomForestClassifier):
        classifier = mod
    else:
        raise ValueError("Invalid model type. Expected RandomForestClassifier or None.")

    # Define pipeline with preprocessing and model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train the model
    model.fit(X, y)'''

    if mod is None:
        filename = "models/import_dependency_None.joblib"
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
def analyze_seasonal_fluctuations(data, trade_type, mod = None):
    import joblib
    import pandas as pd
    from sklearn.svm import SVR
    
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
    if mod == None:
        model = SVR(
    kernel='rbf',      # Radial Basis Function, default kernel
    C=1.0,             # Default regularization
    epsilon=0.1,       # Default margin of tolerance
    gamma='scale')     # Automatic scaling of kernel coefficient

    elif mod == 'seasonal2':
        model = SVR(
    kernel='rbf',      # RBF kernel
    C=10.0,            # Stronger regularization for high complexity
    epsilon=0.2,       # Broader tolerance for errors
    gamma='scale'      # Automatically scaled kernel coefficient
)
    
    elif mod == 'seasonal3':
        model = SVR(
    kernel='poly',     # Polynomial kernel for non-linear patterns
    C=5.0,             # Moderate regularization
    epsilon=0.1,       # Default tolerance for errors
    gamma='auto',      # Use 1/n_features for kernel coefficient
    degree=3           # Default degree for polynomial kernel
)

    
    elif mod == 'seasonal4':
        model = SVR(
    kernel='sigmoid',  # Sigmoid kernel for specific relationships
    C=0.5,             # Lower regularization for simplicity
    epsilon=0.15,      # Moderate tolerance for errors
    gamma='scale'      # Automatically scaled kernel coefficient
)

    
    model.fit(X, y)
    '''if mod is None:
        model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1,
            gamma='scale'
        )
    elif isinstance(mod, SVR):
        model = mod
    else:
        raise ValueError("Invalid model type. Expected SVR or None.")

    # Train the model
    model.fit(X, y)'''
    
    # Save the model and the feature column names used during training
    
    if mod is None:
        filename = f"models/seasonal_{trade_type.lower()}_None.joblib"
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
    
    # Call function 1
    classify_import_dependency(imports_train, exports_train)
    
    # Call function 2 for imports and exports separately
    analyze_seasonal_fluctuations(imports_train, "Imports")
    #analyze_seasonal_fluctuations(exports_train, "Exports")
