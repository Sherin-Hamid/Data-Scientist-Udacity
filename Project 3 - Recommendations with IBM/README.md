# Recommendations with IBM
The third project of the Udacity Data Scientist Nanodegree

## Introduction
For this project, we analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles we think they will like.

## Tasks
The project is divided into the following tasks

### I. Exploratory Data Analysis

Before making recommendations of any kind, we explore the data we are working with for the project. We dive in to see what we can find. There are some basic, required questions to be answered about the data we are working. We use this space to explore, before we dive into the details of our recommendation system.

### II. Rank Based Recommendations

To get started in building recommendations, we first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

### III. User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. We implement this next.

### IV. Matrix Factorization

Finally, we complete a machine learning approach to build recommendations. Using the user-item interactions, we build out a matrix decomposition. Using our decomposition, we get an idea of how well we can predict new articles an individual might interact with.
