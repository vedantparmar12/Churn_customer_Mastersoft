import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template, jsonify
import os

app = Flask(__name__)

# Define the features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge']
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'StreamingBoth', 'StreamingService'
]

def create_model_files():
    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    # Create the full pipeline with the best hyperparameters
    best_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=0.1, penalty='l2', solver='liblinear'))
    ])

    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'tenure': np.random.randint(1, 100, n_samples),
        'MonthlyCharges': np.random.uniform(20, 200, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'AvgMonthlyCharge': np.random.uniform(20, 200, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'StreamingBoth': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingService': np.random.choice(['Yes', 'No'], n_samples),
    })
    y = np.random.choice([0, 1], n_samples)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    best_model.fit(X_train, y_train)

    # Save the fitted model
    model_filename = 'high_accuracy_logistic_regression_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Fitted model saved as {model_filename}")

    # Save the preprocessor separately
    preprocessor_filename = 'preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Preprocessor saved as {preprocessor_filename}")

    # Save the feature names for future use
    feature_names = numeric_features + categorical_features
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Feature names saved as feature_names.pkl")

# Check if model files exist, if not, create them
if not all(os.path.exists(f) for f in ['high_accuracy_logistic_regression_model.pkl', 'preprocessor.pkl', 'feature_names.pkl']):
    create_model_files()

# Load the model and preprocessor
model = joblib.load('high_accuracy_logistic_regression_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Ensure all expected features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = None  # or some default value
        
        # Reorder columns to match the expected order
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]  # Probability of positive class
        
        return jsonify({
            'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    