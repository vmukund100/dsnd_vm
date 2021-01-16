# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys
import pickle

import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
#from nltk import post_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer




def load_data(database_filepath):
    '''
    Description: Load data from sqll table using sqlite
    Arg: file path of the sql table
    Return: X is the variable/data-matrix on which ML will be performed, 
    Y the target and category labels corresponding to each X
    '''
    # load data from database
    engine = create_engine('sqlite:///test.db')
    #df.to_sql('test', engine, index=False)
    df = pd.read_sql_table('test', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    #df.head(5)
    category_labels = list(df.columns[4:])
    return X, Y, category_labels


def tokenize(text):
    '''
    Description: tokenization function to process your text data
    Arg: text
    Return: list of clean tokens after lemmatize, lower case and strip
    '''
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer =  WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Description: Staring verb extractor class (estimator having fit & transform method)
    Will be used in pipeline build
    Arg: estimator & transform component from sklearn.base
    Return: pandas series with verb form of strating word extracted for each message
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

   
def build_model():
    '''
    Description: Build model using multioutputclassifier, feature union and pipelines;
    classifier: AdaBoostClassifier; GridSearch used but takes awfully long time
    Arg: None
    Return: Main ML step using GridSearch; Parameters of cross-validated grid-search or pipeline1 
    '''
    pipeline1 = Pipeline([('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    parameters = {
        'clf__estimator__n_estimators': [10, 20, 40]
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        
        }

    cv = GridSearchCV(pipeline1, param_grid=parameters)

    return cv #pipeline1 


def evaluate_model(model, X_test, Y_test, category_labels):
    '''
    Description: Evaluate model from Pipeline step; 36 categories; iterating through the 36 labels
    Arg: model from pipeline, X - vairable on which ML was performed; Y multioutput targets
    Return: Classification report and accuracy score for each category  
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_labels)):
        print('Category: {}'.format(category_labels[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))
    


def save_model(model, model_filepath):
    '''
    Description: Save the modeled data into pickle file for serilaization
    Arg: model and filepath
    Return: pickle file
    '''
    pickle.dump(model, open('model.pkl', 'wb'))


def main():
    '''
    Description: Class main();Provided code for calling all fuctions defined above for execution;
    Arg: main class
    Return: Print which step/function it is currently executing; 
    Returns dataframe after functions defined above has been executed  
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_labels = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()