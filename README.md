### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The codes runs should run with no issues using python versions 3. Other packages used (not available in Anaconda distribution of python) - Folium (to get beautiful maps given the latitudes and longitudes). The documentation for Folium can be found [here](https://python-visualization.github.io/folium/modules.html#module-folium.map). 

## Project Motivation<a name="motivation"></a>
For this project, I am looking at the Seattle and boston AirBnB dataset from Kaggle found [here](https://www.kaggle.com/airbnb/seattle/notebooks?datasetId=393&sortBy=dateRun). and [here](https://www.kaggle.com/airbnb/boston).

A few questions such as explained below are examined: 

1. what are different property types available in Boston and Seattle? How are availability distribution of most popular property type in each city?

2. How is the pricing distribtuions according to neighbourhoods and property types? what are the pricing movements over the time period available in the data sets?

3. A few key factors affect the price of rental places in Boston and Seattle. We use machine learning model specifically XGBoost regressor to look into most important features affecting the price. 

A minor digression is made by including two feature such as a list of amenities and methods used to verify host cedentials in the dataset. This involves columns of lists/dictionary with elements in quotes. However, it was found that these features does not affect the price significantly. 

## File Descriptions <a name="files"></a>

There are 2 notebooks available here to help walk through the data analysis for answering the questions. There are a couple of picture files to check if they answer the questions at all. The data filenames starting with 'b' such as 'blistings.csv' corresponds to the Boston data set. 

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://vm2018chc2.medium.com/seattle-vs-boston-an-exploration-of-its-airbnb-data-724aa5c9c0b3). 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The data sets are found [here](https://www.kaggle.com/airbnb/seattle/notebooks?datasetId=393&sortBy=dateRun). and [here](https://www.kaggle.com/airbnb/boston). The documentation for Folium can be found [here](https://python-visualization.github.io/folium/modules.html#module-folium.map). The documentation for XG Boost regressor can be found [here](https://xgboost.readthedocs.io/en/latest/index.html). 
