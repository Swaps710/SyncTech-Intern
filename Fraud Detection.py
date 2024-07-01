import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Specify the path to the uploaded Excel file
file_path = 'C:/Users/swapna/OneDrive/Documents/creditcard.csv.xlsx'

# Load the dataset from the Excel file (loading only a subset of rows)
data = pd.read_excel(file_path, nrows=10000)  # Adjust nrows as necessary
print("Dataset loaded successfully from the Excel file.")

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values in the dataset
print("Missing values in each column:\n", data.isnull().sum())

# Basic statistics of the dataset
print("Basic statistics of the dataset:\n", data.describe())

# Check data types of the columns
print("Data types of the columns:\n", data.dtypes)

# Data visualization: Distribution of fraud cases
sns.countplot(x='Class', data=data)
plt.title('Distribution of Fraud Cases')
plt.show()

# Data visualization: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Select relevant features for the model
features = data.columns[:-1]  # Assuming the last column is 'Class'
X = data[features]
y = data['Class']

print("First few rows of features:\n", X.head())
print("First few rows of the target variable:\n", y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model training completed.")

# Make predictions with Random Forest
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Random Forest Accuracy Score:", accuracy_score(y_test, rf_predictions))

# Initialize and train the Stochastic Gradient Descent Classifier
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
print("SGD model training completed.")

# Make predictions with SGD
sgd_predictions = sgd_model.predict(X_test)

# Evaluate SGD model
print("SGD Classification Report:\n", classification_report(y_test, sgd_predictions))
print("SGD Confusion Matrix:\n", confusion_matrix(y_test, sgd_predictions))
print("SGD Accuracy Score:", accuracy_score(y_test, sgd_predictions))

# Initialize and train the Support Vector Classifier
svc_model = SVC(random_state=42)
svc_model.fit(X_train, y_train)
print("SVC model training completed.")

# Make predictions with SVC
svc_predictions = svc_model.predict(X_test)

# Evaluate SVC model
print("SVC Classification Report:\n", classification_report(y_test, svc_predictions))
print("SVC Confusion Matrix:\n", confusion_matrix(y_test, svc_predictions))
print("SVC Accuracy Score:", accuracy_score(y_test, svc_predictions))

# Visualization: Feature importance for Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=features)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances in Random Forest')
plt.show()
