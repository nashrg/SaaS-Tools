from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load pre-trained model (for demonstration purposes, we'll use the same script above)
def load_model():
    # Sample data columns (modify according to your actual data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

model = load_model()

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load the uploaded file into a DataFrame
        df = pd.read_csv(file_path)

        # Data preprocessing (modify according to actual dataset)
        X = df.drop(columns=['customer_id', 'churn'])  # Modify as per the columns in your dataset

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict churn risk using the model
        df['churn_risk'] = model.predict_proba(X_scaled)[:, 1]  # Replace with actual model

        # Get top 10 at-risk customers
        at_risk_customers = df[['customer_id', 'churn_risk']].sort_values(by='churn_risk', ascending=False).head(10)

        # Save results to a new CSV file
        result_file = 'top_at_risk_customers.csv'
        at_risk_customers.to_csv(result_file, index=False)

        # Render the results page with data
        return render_template('results.html', tables=[at_risk_customers.to_html(classes='data')], result_file=result_file)

    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
