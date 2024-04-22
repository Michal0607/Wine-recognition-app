# Wine Recognition App

Wine Recognition App is a simple web application built with Streamlit to demonstrate the process of loading data, training models, and visualizing predictions interactively for wine classification.

## Features

- **Data Visualization**: Visualize the wine dataset directly in your browser, allowing for a hands-on approach to understand the dataset’s structure and distribution.
- **Model Training**: Train multiple classification models including Logistic Regression, Random Forest, and Support Vector Machines (SVM) using the provided wine dataset.
- **Prediction Accuracy**: Evaluate the performance of the trained models through accuracy metrics and visualize the results using confusion matrices.

## Technologies Used

- **Streamlit**: An open-source app framework ideal for rapid development of data-driven applications.
- **Pandas**: For efficient data manipulation and analysis.
- **Scikit-learn**: For implementing robust machine learning models.
- **Matplotlib** and **Seaborn**: Used for creating informative and attractive static and interactive visualization graphs.

## Installation

To run this application locally, you'll need Python, Docker, and Docker Compose installed on your machine.

1. Clone the repository:

   git clone https://github.com/your_username/wine-recognition-app.git

2. Navigate to the project directory:

   cd wine-recognition-app

3. Build and run the Docker containers:
   
   docker-compose up --build

This will start all the services defined in your docker-compose.yml file. You can access the app by navigating to http://localhost:8501/ in your web browser.

## Usage

Once the application is running, navigate to http://localhost:8501/ in your web browser to view and interact with the app.

 - **Show Data**: Interactively view data within the app.
 - **Train Model**: Choose a model to train from the dropdown menu and click the 'Train Model' button to train the model on the wine dataset and view metrics such as accuracy and confusion matrix.
 - **Visualize Data**: Use interactive controls to visualize different aspects of the data and the results of the model predictions.
## Contributing
Contributions are welcome! If you have suggestions for improvements or encounter any issues, please feel free to submit a pull request or open an issue.