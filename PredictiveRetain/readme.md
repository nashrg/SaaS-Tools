1. Directory Structure
   Predictive Retain structure should look like this:

graphql
Copy code
churn-prediction-app/
│
├── app.py
├── uploads/  # Directory for storing uploaded files
│   └── (empty initially)
├── templates/
│   ├── index.html
│   └── results.html
└── customer_data.csv  # Example CSV file for testing

2. Running the Application
   Ensure you have installed Flask:
   bash
   pip install Flask
   Run the Flask app:
   bash
   python app.py
   Open your browser and go to http://127.0.0.1:5000/ to view the application.