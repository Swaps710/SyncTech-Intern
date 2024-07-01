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
