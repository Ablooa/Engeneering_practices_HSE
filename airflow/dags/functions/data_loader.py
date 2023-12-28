import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def load_data(data_path):
    if not os.path.exists(data_path):
        competition_name = 'Titanic - Machine Learning from Disaster'
        files_to_download = ['test.csv', 'train.csv']

        if not os.path.exists('data'):
            os.makedirs('data')

        api = KaggleApi()
        api.authenticate()

        for file in files_to_download:
            api.competition_download_file(competition_name, file, path='data', force=True)

    data = pd.read_csv(data_path)
    return data