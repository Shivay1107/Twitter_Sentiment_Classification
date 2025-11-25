import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ClassicalML:
    def __init__(self, model_name=None, pipeline=None, param_grid=None):
        self.model_name = model_name
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize ML model or return pipeline"""

        # If pipeline is passed → return pipeline directly
        if self.pipeline is not None:
            return self.pipeline
        
        # Otherwise initialize by model name
        if self.model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)

        elif self.model_name == "Random Forest Classifier":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()

        elif self.model_name == "Support Vector Machine":
            from sklearn.svm import SVC
            return SVC(probability=True)

        elif self.model_name == "K-Nearest Neighbors":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier()

        elif self.model_name == "Decision Tree Classifier":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()

        elif self.model_name == "Naive Bayes Classifier":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()

        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def train(self, X_train, y_train):
        """Train model with or without hyperparameter tuning"""

        if self.param_grid:
            grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return accuracy, report, cm

