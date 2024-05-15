# Movie-Recommender-System

![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)
![API](https://img.shields.io/badge/API-TMDB-fcba03)

Movie Roulette is an extensive movie recommendation system that uses machine learning algorithms to tailor movie choices to users based on their likes, watching history, and comments. By monitoring user interactions and movie properties such as genre, actor, and narrative keywords, the system provides personalized suggestions based on individual preferences. Movie Roulette, with its smooth user interface and easy features, offers cinephiles a dynamic and engaging platform for discovering, exploring, and enjoying a curated selection of movies customized to their specific interests.

The details of the movies(title, genre, runtime, rating, poster, etc) are fetched using an API by TMDB, https://www.themoviedb.org/documentation/api, and using the IMDB id of the movie in the API, I did web scraping to get the reviews given by the user in the IMDB site using `beautifulsoup4` and performed sentiment analysis on those reviews.

## Motivation:

The motivation behind the Content-Based Movie Roulette System is to create a sophisticated recommendation system for movie enthusiasts that goes beyond traditional approaches. By leveraging content-based filtering techniques and sentiment analysis, the system aims to provide personalized movie recommendations tailored to individual preferences while also considering the sentiment expressed by users in their feedback. The integration of Asynchronous JavaScript and XML technology enhances the user experience by enabling seamless, asynchronous communication between the client and server, resulting in faster and more responsive interactions.

# Business Value:

## Enhanced User Experience:
The system enhances the user experience by offering personalized movie recommendations based on content similarity and sentiment analysis. This personalized approach fosters user engagement and satisfaction, leading to increased user retention and loyalty.

## Improved Recommendation Accuracy: 
By analyzing movie content and user feedback, the system provides more accurate and relevant recommendations, increasing the likelihood of users discovering movies that align with their tastes and preferences.

## Increased User Interaction: 
The integration of AJAX technology ensures a smooth and interactive browsing experience, allowing users to explore movie recommendations and provide feedback in real-time. This dynamic interaction encourages active participation and fosters a sense of community among users.

# Metrics:
## Precision and Recall: 
Precision measures the proportion of recommended movies that are relevant to the user's preferences, while recall measures the proportion of relevant movies that are successfully recommended. High precision and recall indicate that the system effectively recommends movies that match the user's interests.

## Sentiment Analysis Accuracy: 
The accuracy of sentiment analysis algorithms is evaluated based on their ability to correctly identify and classify the sentiment expressed in user feedback. High accuracy ensures that the system can reliably gauge user sentiment and incorporate it into the recommendation process.

## User Engagement Metrics: 
User engagement metrics, such as the number of active users, session duration, and interaction frequency, provide insights into the level of user engagement with the system. Increasing user engagement indicates that the system is successfully capturing and maintaining user interest.



## How to get the API key?

Create an account in https://www.themoviedb.org/, click on the `API` link from the left hand sidebar in your account settings and fill all the details to apply for API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your `API` sidebar once your request is approved.

## How to run the project?

1. Clone or download this repository to your local machine.
2. Install all the libraries mentioned in the [requirements.txt] file with the command `pip install -r requirements.txt`
3. Get your API key from https://www.themoviedb.org/. (Refer the above section on how to get the API key)
4. Replace YOUR_API_KEY in **both** the places (line no. 15 and 29) of `static/recommend.js` file and hit save.
5. Open your terminal/command prompt from your project directory and run the file `main.py` by executing the command `python main.py`.
6. Go to your browser and type `http://127.0.0.1:5000/` in the address bar.
7. There you go! That's it.

## Architecture

![Recommendation App](https://user-images.githubusercontent.com/36665975/168742738-5435cf76-1a42-4d87-94b4-999e5bfc48d3.png)

## Similarity Score :

How does it decide which item is most similar to the item user likes? Here come the similarity scores.

It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.

## How Cosine Similarity works?

Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.

![image](https://user-images.githubusercontent.com/36665975/70401457-a7530680-1a55-11ea-9158-97d4e8515ca4.png)

### Sources of the datasets

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
3. [List of movies in 2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018)
4. [List of movies in 2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)
5. # [List of movies in 2020](https://en.wikipedia.org/wiki/List_of_American_films_of_2020)

## The APP
<img width="960" alt="app" src="https://github.com/adnanehar/My_Movie_Recommender/assets/140387810/5d058031-4d7f-48a5-b4c4-b13ec23e975c">

<img width="960" alt="app1" src="https://github.com/adnanehar/My_Movie_Recommender/assets/140387810/c151d895-83cc-4f85-86f5-7940931742cd">


### ML System Architecture for Movie Recommendation System

# Overview:

The ML system architecture for the movie recommendation system involves several components working together to provide personalized movie recommendations to users based on their preferences and behavior.

### Components:

# Data Collection:

Collects user interaction data such as movie ratings, searches, and views.
Data sources include user profiles, movie databases, and streaming platforms.

# Data Preprocessing:

Cleans and preprocesses the raw data to remove noise and inconsistencies.
Includes tasks like data cleaning, feature engineering, and normalization.

# Model Training:

Trains machine learning models using preprocessed data.
Models include collaborative filtering, content-based filtering, and hybrid models.
Training may involve techniques like matrix factorization, deep learning, or ensemble methods.

# Model Evaluation:

Evaluates the performance of trained models using metrics like precision, recall, and accuracy.
Conducts offline evaluation using historical data and online evaluation with A/B testing.

# Model Deployment:

Deploys the trained models into a production environment for real-time inference.
May involve containerization using Docker and deployment on cloud platforms like AWS, GCP, or Azure.

# User Interface:

Provides an interface for users to interact with the recommendation system.
Includes web or mobile applications where users can input preferences and receive recommendations.

### System Diagram:

# Key Technologies:

Python for data preprocessing, model training, and deployment.
Libraries like Pandas, NumPy, and Scikit-learn for data manipulation and machine learning.
Frameworks like TensorFlow or PyTorch for deep learning models.
Streamlit or Flask for building user interfaces.
Cloud platforms like AWS, GCP, or Azure for model deployment and scalability.


## Conclusion:

The ML system architecture described above outlines the key components and technologies involved in building a movie recommendation system. By leveraging machine learning techniques and data-driven approaches, the system can deliver personalized recommendations to users, enhancing their movie-watching experience.

### Technologies Used

## Flask (v2.3.2)

Flask is a micro web framework for Python that allows developers to quickly build web applications. It provides a simple and flexible architecture, making it easy to get started with web development. Flask is used as the core framework for building the web application and handling HTTP requests.

## Gunicorn (v19.9.0)

Gunicorn is a Python WSGI HTTP server for Unix. It acts as a reverse proxy server, handling incoming HTTP requests and forwarding them to the Flask application. Gunicorn is used to deploy the Flask application in production environments, providing improved performance and scalability.

## Jinja2

Jinja2 is a modern and designer-friendly templating engine for Python. It is used in conjunction with Flask for rendering HTML templates and generating dynamic content in the web application. Jinja2 allows developers to create reusable templates and easily inject data into HTML pages.

## MarkupSafe (>=2.0)

MarkupSafe is a dependency of Jinja2 that provides HTML markup string escaping and manipulation utilities. It ensures that user input is properly escaped to prevent XSS (cross-site scripting) attacks and other security vulnerabilities.

## Werkzeug (>=2.3.3)

Werkzeug is a WSGI utility library for Python that provides various components for building web applications. It is used by Flask internally for handling HTTP requests and responses, routing, and other web-related tasks.

## numpy (>=1.9.2)

NumPy is a powerful library for numerical computing in Python. It provides support for multidimensional arrays, mathematical functions, linear algebra, and random number generation. NumPy is used in the application for data manipulation and numerical operations.

## scipy (>=0.15.1)

SciPy is a scientific computing library for Python that builds on top of NumPy. It provides additional functionality for optimization, integration, interpolation, signal processing, and other scientific computing tasks. SciPy is used in the application for advanced mathematical and scientific computations.

## nltk (v3.5)

NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet. NLTK is used in the application for natural language processing tasks, such as text tokenization and sentiment analysis.

## scikit-learn (>=0.18)

scikit-learn is a popular machine learning library for Python that provides simple and efficient tools for data mining and data analysis. It features various algorithms for classification, regression, clustering, dimensionality reduction, and more. scikit-learn is used in the application for building machine learning models and performing movie recommendations based on user input.

## pandas (>=0.19)

pandas is a fast, powerful, and flexible data analysis and manipulation library for Python. It provides data structures and functions for working with structured data, such as data frames and series. pandas is used in the application for data preprocessing, manipulation, and analysis.

## beautifulsoup4 (v4.9.1)

Beautiful Soup is a Python library for pulling data out of HTML and XML files. It provides a simple interface for navigating and searching HTML documents, making it easy to extract data from web pages. Beautiful Soup is used in the application for web scraping and extracting movie information from external sources.

## jsonschema (v3.2.0)

jsonschema is a Python library for validating JSON data against a JSON schema. It provides tools for defining and validating JSON schemas, ensuring that JSON data adheres to a specified structure and format. jsonschema is used in the application for validating and verifying JSON data received from external APIs.

## tmdbv3api (v1.6.1)

tmdbv3api is a Python wrapper for The Movie Database (TMDb) API v3. It provides convenient access to TMDb's vast collection of movie data, including movie details, images, trailers, and more. tmdbv3api is used in the application for fetching movie information and metadata from TMDb.

## lxml (v5.2.1)

lxml is a Python library for processing XML and HTML documents. It provides a fast and efficient parser for navigating and manipulating XML/HTML trees. lxml is used in the application for parsing and processing HTML documents during web scraping and data extraction.

## urllib3 (>=1.21.1,<1.26)

urllib3 is a powerful HTTP client for Python that provides features such as connection pooling, SSL/TLS support, and thread safety. It is used in the application for making HTTP requests to external APIs and fetching data from remote servers.

## requests (v2.23.0)

Requests is a simple and elegant HTTP library for Python that allows developers to send HTTP requests and handle responses easily. It provides a high-level interface for making HTTP requests with various HTTP methods, headers, and parameters. Requests is used in the application for sending HTTP requests to external APIs and fetching data from web servers.

## pickleshare (v0.7.5)

pickleshare is a small Python library for persistent storage of Python objects using pickle serialization. It provides a simple interface for storing and retrieving Python objects as files on disk. pickleshare is used in the application for caching and storing machine learning models and other serialized data objects.

## Great Expectations

Great Expectations is an open-source library for validating, documenting, and profiling data. It provides a suite of tools and utilities for defining and enforcing data quality expectations, ensuring that data pipelines and processes produce reliable and trustworthy results.

# Why I Used It

Great Expectations was integrated into the data processing pipeline to define and enforce data quality expectations. By specifying data quality rules and constraints, Great Expectations helped ensure the integrity and consistency of the data used for training machine learning models and making recommendations. Additionally, Great Expectations provided automated data validation and documentation, facilitating transparency and reproducibility in the data pipeline.

## ZenML

ZenML is an open-source MLOps framework for building, running, and managing machine learning pipelines. It provides a high-level abstraction for defining and orchestrating end-to-end machine learning workflows, including data preprocessing, model training, evaluation, and deployment.

# Why I Used It

ZenML was utilized to streamline and automate the machine learning pipeline, from data ingestion to model deployment. Its modular and extensible architecture allowed for easy integration with existing tools and libraries, such as scikit-learn and TensorFlow. By using ZenML, I was able to abstract away the complexities of managing machine learning pipelines and focus on developing and iterating on machine learning models more efficiently.

## MLflow

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools and utilities for experiment tracking, model packaging, deployment, and model management. MLflow enables organizations to track experiments, reproduce results, and deploy models to production with ease.

# Why I Used It

MLflow was employed for experiment tracking and model management throughout the machine learning lifecycle. By logging parameters, metrics, and artifacts during model training and evaluation, MLflow facilitated reproducibility and collaboration among team members. Additionally, MLflow's model packaging and deployment capabilities streamlined the process of deploying trained models to production environments, ensuring consistency and reliability in model deployments.

## Docker

Docker is a containerization platform that allows developers to package applications and dependencies into lightweight, portable containers. Containers encapsulate the application and its dependencies, ensuring consistency and reproducibility across different environments.

# Why I Used It

Docker was utilized to containerize the web application and its dependencies, including the Flask application, machine learning models, and external libraries. By containerizing the application, I was able to create a consistent and isolated environment for running the application, regardless of the underlying infrastructure. Docker also facilitated deployment and scaling of the application, making it easier to manage and maintain in production environments.
