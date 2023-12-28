from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


def preprocess_data(file_path, target):
    data = pd.read_csv(file_path)
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    scaler = StandardScaler()

    num_columns = ['Age', 'SibSp', 'Parch', 'Fare', 'Survived' if target == 'Pclass' else 'Pclass']
    cat_columns = ['Sex', 'Embarked']

    X_train, X_test, y_train, y_test = train_test_split(data, data[target], test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', num_imputer), ('scaler', scaler)]), num_columns),
            ('cat', one_hot_encoder, cat_columns)
        ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print(X_train.shape)
    print(X_test.shape)
    
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    
    X_train_df.to_csv('X_train.csv', index=False)
    X_test_df.to_csv('X_test.csv', index=False)
    y_train_df.to_csv('y_train.csv', index=False)
    y_test_df.to_csv('y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


