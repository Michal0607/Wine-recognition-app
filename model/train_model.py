import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def train_and_save_model(choice, data_path, model_path):
    data = pd.read_csv(data_path)
    
    if 'label' not in data.columns:
        raise ValueError("Column 'label' not found in the data")

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if choice == 'logistic_regression':
        model = LogisticRegression()
    elif choice == 'random_forest':
        model = RandomForestClassifier()
    elif choice == 'svm':
        model = SVC()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    joblib.dump(model, model_path)

    return accuracy, cm
