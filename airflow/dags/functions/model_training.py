from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import joblib
import pandas as pd


def train(model_type, base_path):
    X_train_path = base_path + 'X_train.csv'
    y_train_path = base_path + 'y_train.csv'
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    if model_type == 'Логистическая регрессия':
        param_grid_lr = {
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [1000, 10000],
            'penalty': ['l2']
        }
        grid_lr = GridSearchCV(
            LogisticRegression(solver='lbfgs'),
            param_grid_lr,
            cv=5,
            scoring='accuracy'
        )
        grid_lr.fit(X_train, y_train)
        print("Логистическая регрессия:", grid_lr.best_params_, grid_lr.best_score_)
        best_lr_model = grid_lr.best_estimator_
        joblib.dump(best_lr_model, 'best_logistic_regression_model.pkl')
        return best_lr_model

    elif model_type == 'Дерево решений':
        param_grid_dt = {
            'max_depth': [3, 5, 10, 15],
            'min_samples_split': [2, 5, 10, 15]
        }
        grid_dt = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid_dt,
            cv=5,
            scoring='accuracy'
        )
        grid_dt.fit(X_train, y_train)
        print("Дерево решений:", grid_dt.best_params_, grid_dt.best_score_)
        best_dt_model = grid_dt.best_estimator_
        joblib.dump(best_dt_model, 'best_decision_tree_model.pkl')
        return best_dt_model

    elif model_type == 'SVM':
        param_grid_svm = {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        grid_svm = GridSearchCV(
            SVC(),
            param_grid_svm,
            cv=5,
            scoring='accuracy'
        )
        grid_svm.fit(X_train, y_train)
        print("SVM:", grid_svm.best_params_, grid_svm.best_score_)
        best_svm_model = grid_svm.best_estimator_
        joblib.dump(best_svm_model, 'best_svm_model.pkl')
        return best_svm_model

    else:
        print(f"Неизвестный тип модели: {model_type}")

