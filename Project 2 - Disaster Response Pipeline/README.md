# Disaster Response Pipeline Project

## Project Overview:
In this project, I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the project data folder, you'll find a dataset containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the user can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Project Components:
There are three components in this project.

### 1. ETL Pipeline
The Python script, process_data.py, includes a data cleaning pipeline that:
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

### 2. ML Pipeline
The Python script, train_classifier.py, includes a machine learning pipeline that:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

### 3. Flask Web App
A web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


## Run Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To run the Flask Web app
        1. Open a new terminal window. Use terminal commands to navigate inside the app folder with the run.py                file.
        2. Type in the command line: 'python run.py'
           
           The web app should now be running if there were no errors.
        4. Now, open another Terminal Window.
        5. Type: env|grep WORK
        6. In a new web browser window, type in the following: 
           https://SPACEID-3001.SPACEDOMAIN
           You should be able to see the web app. The number 3001 represents the port where your web app will show            up. Make sure that the 3001 is part of the web address you type in.
