from sklearn.utils import shuffle
import os
import pandas as pd

def split_data(import_csv, export_csv):
    imports = pd.read_csv(import_csv)
    exports = pd.read_csv(export_csv)

    shuffle_imports = shuffle(imports)
    shuffle_exports = shuffle(exports)

    samples_im = len(shuffle_imports)
    training_im = int(samples_im * 0.8)
    import_train = shuffle_imports[:training_im]
    import_test = shuffle_imports[training_im:]

    samples_ex = len(shuffle_exports)
    training_ex = int(samples_ex * 0.8)
    export_train = shuffle_exports[:training_ex]
    export_test = shuffle_exports[training_ex:]

    import_train.to_csv(os.path.join('data_processed', 'import_train.csv'), index=False)
    import_test.to_csv(os.path.join('data_processed', 'import_test.csv'), index=False)
    export_train.to_csv(os.path.join('data_processed', 'export_train.csv'), index=False)
    export_test.to_csv(os.path.join('data_processed', 'export_test.csv'), index=False)

def train_and_predict():
    from wf_ml_training import classify_import_dependency, analyze_seasonal_fluctuations
    from wf_ml_prediction import predict_import_dependency, predict_seasonal_fluctuations

    imports_train = pd.read_csv("data_processed/import_train.csv")
    exports_train = pd.read_csv("data_processed/export_train.csv")
    imports_test = pd.read_csv("data_processed/import_test.csv")
    exports_test = pd.read_csv("data_processed/export_test.csv")

    # Train and predict default models
    classify_import_dependency(imports_train, exports_train, 'dependency1')
    analyze_seasonal_fluctuations(imports_train, "Imports", 'seasonal1')
    
    predict_import_dependency(imports_test, exports_test, 'dependency1')
    predict_seasonal_fluctuations(imports_test, "Imports", 'seasonal1')

def alternative_models():
    from wf_ml_training import classify_import_dependency, analyze_seasonal_fluctuations
    from wf_ml_prediction import predict_import_dependency, predict_seasonal_fluctuations

    imports_train = pd.read_csv("data_processed/import_train.csv")
    exports_train = pd.read_csv("data_processed/export_train.csv")
    imports_test = pd.read_csv("data_processed/import_test.csv")
    exports_test = pd.read_csv("data_processed/export_test.csv")

    dependency_models = ['dependency2', 'dependency3', 'dependency4']
    seasonal_models = ['seasonal2', 'seasonal3', 'seasonal4']

    for i, dep_model in enumerate(dependency_models, start=2):
        classify_import_dependency(imports_train, exports_train, mod=dep_model)
        predict_import_dependency(imports_test, exports_test, mod=f"dependency{i}")

    for i, sea_model in enumerate(seasonal_models, start=2):
        analyze_seasonal_fluctuations(imports_train, "Imports", mod=sea_model)
        predict_seasonal_fluctuations(imports_test, "Imports", mod=f"seasonal{i}")

def model_evaluation():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

    dependency_results = {}
    seasonal_results = {}

    dependency_models = ['dependency1', 'dependency2', 'dependency3', 'dependency4']
    seasonal_models = ['seasonal1', 'seasonal2', 'seasonal3', 'seasonal4']

    for model in dependency_models:
        import_results = pd.read_csv(f"evaluation/import_dependency_predictions_{model}.csv")

        y_true_import = import_results['Encoded Label']
        y_pred_import = import_results['Predicted Label']

        label_mapping = {"Low": 0, "Medium": 1, "High": 2}
        y_true_import = y_true_import.map(label_mapping).fillna(0)
        y_pred_import = y_pred_import.map(label_mapping).fillna(0)

        dependency_results[model] = {
            'Accuracy': round(accuracy_score(y_true_import, y_pred_import),8),
            'F1 Score': round(f1_score(y_true_import, y_pred_import, average='weighted'),8)
        }

    for model in seasonal_models:
        seasonal_data = pd.read_csv(f"evaluation/seasonal_imports_predictions_{model}.csv")

        y_true_seasonal = seasonal_data['Dollar value']
        y_pred_seasonal = seasonal_data['Predicted Dollar Value']

        rmse = np.sqrt(mean_squared_error(y_true_seasonal, y_pred_seasonal))
        r2 = r2_score(y_true_seasonal, y_pred_seasonal)

        seasonal_results[model] = {
            'RMSE': round(rmse,8),
            'R-squared': round(r2,8)
        }

    dependency_df = pd.DataFrame.from_dict(dependency_results, orient='index')
    seasonal_df = pd.DataFrame.from_dict(seasonal_results, orient='index')

    with open("evaluation/summary.txt", "w") as f:
        f.write("Dependency Models Evaluation:\n")
        f.write(dependency_df.to_string())
        f.write("\n\nSeasonal Models Evaluation:\n")
        f.write(seasonal_df.to_string())

    print("Evaluation metrics have been saved to 'evaluation/summary.txt'")

if __name__ == "__main__":
    import_data = os.path.join("data_processed","Cleaned_Imports.csv")
    export_data = os.path.join("data_processed","Cleaned_Exports.csv")

    split_data(import_data, export_data)
    train_and_predict()
    alternative_models()
    model_evaluation()



    
