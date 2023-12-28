from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib


def validate_model(name, base_path, model_name):
    X_test_path = base_path + 'X_test.csv'
    y_test_path = base_path + 'y_test.csv'
    model_path = base_path + model_name    
    
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    model = joblib.load(model_path)
    
    y_pred = model.predict(X_test)

    print(f"Модель: {name}")
    print(f"Точность: {accuracy_score(y_test, y_pred)}")
    print(f"Классификационный отчет:\n{classification_report(y_test, y_pred)}")
    print(f"Матрица ошибок:\n{confusion_matrix(y_test, y_pred)}\n")
