import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

def split_data(import_csv, export_csv):
    import_csv = os.path.join("data_processed", "Cleaned_Imports.csv")
    export_csv = os.path.join("data_processed", "Cleaned_Exports.csv")

    imports = pd.read_csv(import_csv)
    exports = pd.read_csv(export_csv)

    shuffle_imports = shuffle(imports, random_state=42)
    shuffle_exports = shuffle(exports, random_state=42)

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
    with open("wf_ml_training.py") as trainer:
        exec(trainer.read())

    with open("wf_ml_prediction.py") as predictor:
        exec(predictor.read())

def model_evaluation():
    #code to evaluate the model based on predictions
    x = 2

def alternative_models():
    #code for the alternative models possible
    x = 2

if __name__ == "__main__":
    import_data = os.path.join("data_processed", "Cleaned_Imports.csv")
    export_data = os.path.join("data_processed", "Cleaned_Exports.csv")

    split_data(import_data, export_data)
    train_and_predict()
    #model_evaluation()
    #alternative_models()



    
