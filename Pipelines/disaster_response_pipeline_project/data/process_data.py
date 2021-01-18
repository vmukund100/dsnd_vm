# Import requirements; Note sqlalchemy
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Description: Load/Extract data from CSV file to a pandas.DataFrame
    Arg: file path of the CSV file
    Return: Returns DataFrame/data-matrix 
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = df_messages.merge(df_categories, how = 'outer', on= 'id')
    #print('files merged')
    return df

def clean_data(df):
    '''
    Description: All cleaning operations performed in ETL_pipeline_preparation.IPYNB
    Arg: pandas.DataFrame created in load_data fucntion
    Return: Returns DataFrame/data-matrix after transformation/cleaning
    '''
    # create a dataframe of the 36 individual category columns
    df2_categories = df['categories'].str.split(';', expand =True)
    #select the first row of the categories dataframe and use this row to extract a list of new column names for categories.
    row = df2_categories.loc[0]
    category_colnames = row.apply(lambda x:x[:-2]).values.tolist()
    df2_categories.columns = category_colnames
    df2_categories.related.loc[df2_categories.related=='related-2'] = 'related-1'
    df2_categories.columns = category_colnames
    # convert category values to just 1 or 0
    for column in df2_categories:
        # set each value to be the last character of the string
        df2_categories[column] = df2_categories[column].str[-1]
        # convert column from string to numeric
        df2_categories[column] = pd.to_numeric(df2_categories[column])
    # drop the original categories column from `df`   
    df = df.drop('categories', axis =1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, df2_categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Description: Load the transformed data into an SQL database for ML
    Arg: pandas.DataFrame transformed in clean_data
    Return: Saves .db file 
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('test', engine, index=False,  if_exists='replace')
    #engine.dispose()
     


def main():
    '''
    Description: Class main();Provided code for calling all fuctions defined above for execution;
    Arg: main class
    Return: Print which stap/function it is currently executing; 
    Returns dataframe after functions defined above has been executed  
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
