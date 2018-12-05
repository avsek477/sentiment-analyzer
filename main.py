import classify_data as get_raw_data
import preprocessor as pre

from sklearn.linear_model import SGDClassifier
import numpy as np
from os import path
import pandas as pd
import classification as clf

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

ORIGINAL_FILE_PATH = "./data/imdb_master.csv"
TRAIN_DATA_FILE_PATH = "./data/train.csv"
TEST_DATA_FILE_PATH = "./data/test.csv"
CLASSIFIER_MODELS = ["1. Logistic Regression", "2. Naive Bayers", "3. DecisionTree Classifier", "4. SGD Classifier"]

def generate_clean_data():
    train_data, test_data = get_raw_data.classify_data(ORIGINAL_FILE_PATH)
    print("The data is being processed right now. Please wait....")
    train_data = pre.preprocess_data(train_data)
    train_data.to_csv("./data/train.csv", sep=",", encoding="utf-8", index=False)
    test_data = pre.preprocess_data(test_data)
    test_data.to_csv("./data/test.csv", sep=",", encoding="utf-8", index=False)
    use_classifier()

def get_user_selected_model(user_choice):
    # switcher = {}
    # for model in CLASSIFIER_MODELS:
    #     item = model.split(".")
    #     switcher[item[0]]=item[1].strip()
    switcher = {
        "1": LogisticRegression(C=100),
        "2": MultinomialNB(fit_prior=False),
        "3": DecisionTreeClassifier(),
        "4": SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)
    }
    return switcher.get(user_choice, False)
    
def use_classifier():
    train_data = pd.read_csv(TRAIN_DATA_FILE_PATH, encoding="utf-8").sample(frac=1).reset_index(drop=True)
    test_data = pd.read_csv(TEST_DATA_FILE_PATH, encoding="utf-8").sample(frac=1).reset_index(drop=True)
    for model in CLASSIFIER_MODELS:
        print(model)
    model_input = input("Choose a classfier model to train the data[1-" + str(len(CLASSIFIER_MODELS)) + "]:")
    model = get_user_selected_model(model_input)
    print(model)
    if(model != False):
        print("Training the selected classifier model...")
        classifier = clf.Classification(model, train_data, test_data)
        accuracy, matrix = classifier.train_classifier_model()
        print("The model predicts", accuracy*100, "%", "ACCURATELY")
        print("CONFUSION MATRIX for this model is: ", "\n", matrix)
        prediction_choice = input("Would you like to enter your review for the model to predict[yes/no]?")
        if(prediction_choice.lower() == "y" or prediction_choice.lower() == "yes"):
            prediction_input = input("Enter your review for prediction:")
            print("POSITIVE" if classifier.predict_text(pre.stemma(prediction_input))[0] == 1 else "NEGATIVE")
        else:
            print("THANK YOU FOR YOUR TIME.")
            return
    else:
        print("There is no such model specified in the option above. Please try again.")
        return
    
def main():
    if(path.exists(TRAIN_DATA_FILE_PATH) and path.exists(TEST_DATA_FILE_PATH)):
        if(path.getsize(TRAIN_DATA_FILE_PATH)>0 and path.getsize(TEST_DATA_FILE_PATH)):
            use_classifier()
        else:
            generate_clean_data()
    else:
        generate_clean_data()

if __name__ == "__main__":
    main()