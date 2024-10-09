# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Explore the dataset
print(df.head())
print(df.info())

# Data preprocessing
# Check for missing values and fill them (forward fill as an example, use better methods for numeric values)
df.fillna(method='ffill', inplace=True)

# Encoding categorical variables (if applicable)
# Example: df = pd.get_dummies(df, drop_first=True)

#drop unnecessary columns
X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

# Check class imbalance
print(f"Churn value counts:\n{y.value_counts()}")

# Split the data into training and testing sets, stratify to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights to handle imbalance (if necessary)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Model building: Random Forest Classifier with class weights
rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize the feature importances
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title('Top 10 Feature Importances')
plt.show()

# Visualize the confusion matrix with better labeling
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Conclusion: Use the model to identify at-risk customers
df['churn_risk'] = best_rf_model.predict_proba(scaler.transform(X))[:, 1]  # Predict churn probability

# Show top 10 at-risk customers
at_risk_customers = df[['customer_id', 'churn_risk']].sort_values(by='churn_risk', ascending=False).head(10)
print('Top 10 At-Risk Customers:')
print(at_risk_customers)

# Save the results to a CSV file
at_risk_customers.to_csv('at_risk_customers.csv', index=False)
