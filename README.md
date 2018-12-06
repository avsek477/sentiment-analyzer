# sentiment-analysis

Little context on the project

1. Train and Test Data: The original source of the data was downloaded from kaggle and the data represents a list of positive and negative IMBD reviews. However, the list also contains a bunch of unclassified positive and negative reviews labelled as unsup. There was no other context in the data specified so as to classify the unsup labelled data so, I thought it would be best if I were to discard the unclassified reviews to produce a better prediction model. The above mentioned things are in the coded form in the file classify_data.py

2. preprocessor.py: The objective of this file is to preprocess the classified train and test data that we get from classified.py . The method inside this file first encodes the positive and negative reviews into ones and zeros where 1 is the positive review and 0 is the negative review. Also, the preprocess_data function removes any punctuation marks detected using regex expressions which can be seen in the REPLACE_WITH_NO_SPACE variable and, the html tags detected are also removed and replaced with spaces as seen in the REPLACE_WITH_SPACE variable. Finally, each words in the reviews are broken down to its root form avoiding stop words using Snowball Stemmer Algorithm. The reason I used this stemmer is because it is an improvement over porter Stemmer which provides slighly faster computation time as well. Also, this algorithm isnt that much of an aggressive preventing on changing the words to fault. 

3. classification.py : This file contains of a class and methods in it to train different classifier models and predict the user inputted review as selected by the user in the terminal. The class can be instantiated by passing in the model that they want to train and providing the preprocessed train and test data.

4. main.py : Code where all of the magic happens. All of the things needed to understand about this file is shown in the terminal. Please run this file after unzipping the imdb_review data located inside the "data" folder.

#STEPS TO FOLLOW:

Step 1: Go to the data folder and unzip the csv format.

Step 2: Go to the root folder and run the main.py file.

P.S. When you run the main.py for the first it takes time to create a clean train.csv and test.csv file. After the files are created, follow the steps as shown in the terminal UI.