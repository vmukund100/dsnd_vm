### Disaster Response Pipeline Project
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Files Used and description](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

The codes runs should run with no issues using python versions 3. Other used Python Libraries used: pandas, sklearn, sqlite3, sqlalchemy, nltk, plotly, flask, HTML, Bootstrap. This project also uses NLP packages, multioutputclassifier and AdaBoost Classifier to categorize messages sent in the wake of a disaster. 

## Project Motivation<a name="motivation"></a>
A web app is being implemented for an emergency worker to input a new message and figure out which all catgeories the new message fall within. Categorising a message will help in alerting different aid provinding departments such as Police department or hospitals depending on the circumstances. The data was provided by Figure Eight,Inc now acquired by [Appen](https://appen.com/)

## Files used and description <a name="files"></a>
There is a small heirarchy of short codes here. The codes are built in three steps - ETL pipeline was built in process.py, ML pipeline was built in train_process.py and finally the app is run from run.py. There are 3 folder - 
1. app - (a) template contains go.html and master.html (b) run.py
2. data - disaster_categories.csv, disaster_messages.csv, process_data.py and a .db file created
3. models - train_classifier.py and classifier.pkl

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. Input a message. For example: We are 15 people stranded on Euclid Ave. We lost power and have heavy floods here. Not sure when the rescue team will be here. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Documentation of multioutput classifier can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) and that of cross validated griad search can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
