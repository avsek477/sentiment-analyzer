import pandas as pd

def classify_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
    data = data[data['label'] != "unsup"]
    selected_column = ['review', 'label']
    train_data = data[data['type'] == "train"].copy()
    test_data = data[data['type'] == "test"].copy()
    return train_data[selected_column], test_data[selected_column]