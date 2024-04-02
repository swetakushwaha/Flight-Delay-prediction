# -*- coding: utf-8 -*-

# Name: Sweta Kushwaha
# Date: 03/18/2024
# AI-539 (Machine Learning Challenges)
# Final Implementation

"""# IMPORT"""

import pandas as pd
pd.set_option('display.max_columns', 30)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew, boxcox, yeojohnson
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import warnings
warnings.filterwarnings('ignore')

"""# GET DATA"""

# load data
data = pd.read_csv("flightdata.csv")

"""# EDA"""

# Get the data types of each column
feature_types = data.dtypes

# Print the feature types
print("Feature Types:")
print(feature_types)

data.head()

data.describe()

data.columns

data.shape

# Number of Items
len(data)

# Number of Classes
data['ARR_DEL15'].nunique()
print(data['ARR_DEL15'].nunique())
print(data['ARR_DEL15'].dtype)

numerical_columns = data.select_dtypes(include=['number']).columns
numerical_columns = [col for col in numerical_columns if col != 'Unnamed: 25']
print(numerical_columns)

categorical_columns = data.select_dtypes(include=['object']).columns
print(categorical_columns)

for column in categorical_columns:
    unique_values = data[column].nunique()
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, order=data[column].value_counts().index)
    plt.title(f'Histogram of Distinct Values for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    print(f"\nNumber of Unique Values for {column}: {unique_values}\n")
    plt.show()

for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Count occurrences of each class
class_counts = data['ARR_DEL15'].value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of ARR_DEL15')
plt.xlabel('ARR_DEL15')
plt.ylabel('Count')
plt.xticks([0, 1], ['On Time (0)', 'Delayed (1)'], rotation=0)
plt.show()

# Calculate class ratios
class_ratios = class_counts / len(data)

print(class_ratios)

# Count occurrences of each class
class_counts = data['DEP_DEL15'].value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of DEP_DEL15')
plt.xlabel('DEP_DEL15')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Delayed (0)', 'Delayed (1)'], rotation=0)
plt.show()

# Count occurrences of each class for the "CANCELLED" column
cancelled_counts = data['CANCELLED'].value_counts()

plt.figure(figsize=(6, 4))
cancelled_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Cancelled Flights')
plt.xlabel('CANCELLED')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Cancelled (0)', 'Cancelled (1)'], rotation=0)
plt.show()

# Count occurrences of each class
class_counts = data['DIVERTED'].value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of DIVERTED')
plt.xlabel('DIVERTED')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Diverted (0)', 'Diverted (1)'], rotation=0)
plt.show()

"""# SPLIT"""

# Drop irrelevant columns
data = data.drop(['YEAR', 'UNIQUE_CARRIER','Unnamed: 25'], axis=1)

# data = data.dropna(subset=["ARR_DEL15"])

X = data.drop('ARR_DEL15', axis=1)
y = data['ARR_DEL15']

# Extract first two digits of "TAIL_NUM" directly in the same column
X['TAIL_NUM'] = X['TAIL_NUM'].astype(str).str[:2]

# One-hot encode categorical columns
categorical_columns = ['ORIGIN', 'DEST', 'TAIL_NUM']
X = pd.get_dummies(X, columns=categorical_columns)

# Split data into train and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Further split train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2

"""# BASELINE with train-test split"""

X_TR = X_train.copy()
y_TR = y_train.copy()
X_VA = X_val.copy()
y_VA = y_val.copy()
X_TS = X_test.copy()
y_TS = y_test.copy()

# Drop rows with missing values
X_TR.dropna(inplace=True)
y_TR = y_TR[X_TR.index]
X_VA.dropna(inplace=True)
y_VA = y_VA[X_VA.index]
X_TS.dropna(inplace=True)
y_TS = y_TS[X_TS.index]

# Predict the most frequent class on the training set
most_frequent_class = np.bincount(y_TR.astype(int)).argmax()
y_TR_pred = np.full_like(y_TR, fill_value=most_frequent_class)

# Predict the most frequent class on the validation set
y_VA_pred = np.full_like(y_VA, fill_value=most_frequent_class)

# Predict the most frequent class on the testing set
y_TS_pred = np.full_like(y_TS, fill_value=most_frequent_class)

# Evaluate metrics on training set
train_precision = precision_score(y_TR, y_TR_pred, pos_label=most_frequent_class)

# Evaluate metrics on validation set
val_precision = precision_score(y_VA, y_VA_pred, pos_label=most_frequent_class)


# Evaluate metrics on testing set
test_precision = precision_score(y_TS, y_TS_pred, pos_label=most_frequent_class)

# Print baseline performance
print("TRAINING SET PERFORMANCE (Baseline)")
print("Precision: {}".format(round(train_precision, 4)))
print("\nVALIDATION SET PERFORMANCE (Baseline)")
print("Precision: {}".format(round(val_precision, 4)))
print("\nTESTING SET PERFORMANCE (Baseline)")
print("Precision: {}".format(round(test_precision, 4)))

"""# HANDLING SKEWNESS"""

# List of classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Logistic Regression": LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=100),
    "Random Forest": RandomForestClassifier(random_state=0)
}

# Dataframe to store results
results_df = pd.DataFrame(columns=['Method'] + list(classifiers.keys()))

# Define skewness handling methods
skewness_methods = ["boxcox", "yeojohnson", "reciprocal"]

# Iterate over skewness handling methods
for method in skewness_methods:

    X_TR = X_train.copy()
    y_TR = y_train.copy()
    X_VA = X_val.copy()
    y_VA = y_val.copy()

    index_TR = X_TR.dropna().index
    index_VA = X_VA.dropna().index
    X_TR = X_TR.loc[index_TR]
    y_TR = y_TR[index_TR]
    X_VA = X_VA.loc[index_VA]
    y_VA = y_VA[index_VA]

    # Lists to store evaluation metrics
    precision_dict = {}

    # skewness of relevant columns
    relevant_columns = ['DEP_DELAY', 'ARR_DELAY', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'DISTANCE']

    # Apply skewness transformation to relevant columns
    for col in relevant_columns:
        if method == "boxcox":
            X_TR[col], _ = boxcox(X_TR[col] - X_TR[col].min() + 1)  # Shift values to avoid negative
            X_VA[col], _ = boxcox(X_VA[col] - X_VA[col].min() + 1)  # Shift values to avoid negative
        elif method == "yeojohnson":
            X_TR[col], _ = yeojohnson(X_TR[col])
            X_VA[col], _ = yeojohnson(X_VA[col])
        elif method == "reciprocal":
            X_TR[col] = 1 / (X_TR[col] + 1e-6)  # Adding a small value to avoid division by zero
            X_VA[col] = 1 / (X_VA[col] + 1e-6)

    # Clip outliers
    for col in ['DEP_DELAY', 'DEP_DEL15', 'ARR_DELAY', 'CANCELLED', 'DIVERTED','ACTUAL_ELAPSED_TIME']:
        lower_threshold = X_TR[col].mean() - 3 * X_TR[col].std()
        upper_threshold = X_TR[col].mean() + 3 * X_TR[col].std()

        # Clip outliers in the training set
        X_TR[col] = np.clip(X_TR[col], lower_threshold, upper_threshold)

        # Clip outliers in the val set
        X_VA[col] = np.clip(X_VA[col], lower_threshold, upper_threshold)

    for clf_name, clf in classifiers.items():
        # Initialize and train classifier
        clf.fit(X_TR, y_TR)

        # Predict on the test set
        y_pred = clf.predict(X_VA)

        # Calculate precision
        precision = round(precision_score(y_VA, y_pred), 4)
        precision_dict[clf_name] = precision

    # Append precision results to the dataframe
    precision_dict['Method'] = method
    results_df = results_df.append(precision_dict, ignore_index=True)

# Sort the results dataframe by Method in ascending order
results_df = results_df.sort_values(by='Method', ascending=True)

# Print the results dataframe
print("FINDING THE BEST METHOD TO HANDLE SKEWNESS AND OUTLIERS")
print(results_df)

results_df

"""# HANDLING SCALING"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier


X_TR = X_train.copy()
y_TR = y_train.copy()
X_VA = X_val.copy()
y_VA = y_val.copy()

index_TR = X_TR.dropna().index
index_VA = X_VA.dropna().index
X_TR = X_TR.loc[index_TR]
y_TR = y_TR[index_TR]
X_VA = X_VA.loc[index_VA]
y_VA = y_VA[index_VA]

# handling skewness
relevant_columns = ['DEP_DELAY', 'ARR_DELAY', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'DISTANCE']
for col in relevant_columns:
  X_TR[col] = 1 / (X_TR[col] + 1e-6)
  X_VA[col] = 1 / (X_VA[col] + 1e-6)

# handling outliers
for col in ['DEP_DELAY', 'DEP_DEL15', 'ARR_DELAY', 'CANCELLED', 'DIVERTED','ACTUAL_ELAPSED_TIME']:
        lower_threshold = X_TR[col].mean() - 3 * X_TR[col].std()
        upper_threshold = X_TR[col].mean() + 3 * X_TR[col].std()

        # Clip outliers in the training set
        X_TR[col] = np.clip(X_TR[col], lower_threshold, upper_threshold)

        # Clip outliers in the val set
        X_VA[col] = np.clip(X_VA[col], lower_threshold, upper_threshold)

# Strategies to handle scaling
scalers = {
    "None": None,
    "MinMax": MinMaxScaler(feature_range=(0, 1)),
    "Standard": StandardScaler()
}

# List of classifiers
classifiers = {
          "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
          "Logistic Regression": LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=100),
          "Random Forest": RandomForestClassifier(random_state=0)
        }

results_df = pd.DataFrame(columns=['Method'] + list(classifiers.keys()))


for name, scaler in scalers.items():
        if name == "None":
          pass
        else:
          scaler.fit(X_TR)
          X_TR = scaler.transform(X_TR)
          X_VA = scaler.transform(X_VA)

        for clf_name, clf in classifiers.items():
          # Initialize and train classifier
          clf.fit(X_TR, y_TR)

          # Predict on the test set
          y_pred = clf.predict(X_VA)

          # Calculate precision
          precision = round(precision_score(y_VA, y_pred), 4)
          precision_dict[clf_name] = precision

        # Append precision results to the dataframe
        precision_dict['Method'] = name
        results_df = results_df.append(precision_dict, ignore_index=True)

# Sort the results dataframe by Method in ascending order
results_df = results_df.sort_values(by='Method', ascending=True)

# Print the results dataframe
print("FINDING THE BEST METHOD TO HANDLE SCALING")
print(results_df)

"""# HANDLING CLASS IMBALANCE"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier


X_TR = X_train.copy()
y_TR = y_train.copy()
X_VA = X_val.copy()
y_VA = y_val.copy()

index_TR = X_TR.dropna().index
index_VA = X_VA.dropna().index
X_TR = X_TR.loc[index_TR]
y_TR = y_TR[index_TR]
X_VA = X_VA.loc[index_VA]
y_VA = y_VA[index_VA]

# handling skewness
relevant_columns = ['DEP_DELAY', 'ARR_DELAY', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'DISTANCE']
for col in relevant_columns:
    X_TR[col] = 1 / (X_TR[col] + 1e-6)
    X_VA[col] = 1 / (X_VA[col] + 1e-6)

# handling outliers
for col in ['DEP_DELAY', 'DEP_DEL15', 'ARR_DELAY', 'CANCELLED', 'DIVERTED','ACTUAL_ELAPSED_TIME']:
    lower_threshold = X_TR[col].mean() - 3 * X_TR[col].std()
    upper_threshold = X_TR[col].mean() + 3 * X_TR[col].std()

    # Clip outliers in the training set
    X_TR[col] = np.clip(X_TR[col], lower_threshold, upper_threshold)

    # Clip outliers in the val set
    X_VA[col] = np.clip(X_VA[col], lower_threshold, upper_threshold)

# List of classifiers
classifiers = {
          "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
          "Logistic Regression": LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=100),
          "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=20, random_state=0)
        }

results_df = pd.DataFrame(columns=['Method'] + list(classifiers.keys()))

# Strategies to handle class imbalance
methods = ["undersampling", "oversampling", "smote"]

for method in methods:
        if method == "undersampling":
            sampler = RandomUnderSampler(sampling_strategy='auto', random_state=0)
            X_TR, y_TR = sampler.fit_resample(X_TR, y_TR)

        elif method == "oversampling":
            sampler = RandomOverSampler(sampling_strategy='auto', random_state=1)
            X_TR, y_TR = sampler.fit_resample(X_TR, y_TR)

        elif method == "smote":
            sampler = SMOTE(sampling_strategy='auto', random_state=2)
            X_TR, y_TR = sampler.fit_resample(X_TR, y_TR)

        for clf_name, clf in classifiers.items():
          # Initialize and train classifier
          clf.fit(X_TR, y_TR)

          # Predict on the test set
          y_pred = clf.predict(X_VA)

          # Calculate precision
          precision = precision_score(y_VA, y_pred)
          precision_dict[clf_name] = precision

        # Append precision results to the dataframe
        precision_dict['Method'] = method
        results_df = results_df.append(precision_dict, ignore_index=True)

# Print the results dataframe
print("FINDING THE BEST METHOD TO HANDLE CLASS IMBALANCE")
print(results_df)

"""# TEST SET PREDICTION"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier


X_TR = X_train.copy()
y_TR = y_train.copy()
X_VA = X_val.copy()
y_VA = y_val.copy()
X_TS = X_val.copy()
y_TS = y_val.copy()

index_TR = X_TR.dropna().index
index_VA = X_VA.dropna().index
index_TS = X_VA.dropna().index
X_TR = X_TR.loc[index_TR]
y_TR = y_TR[index_TR]
X_VA = X_VA.loc[index_VA]
y_VA = y_VA[index_VA]
X_TS = X_TS.loc[index_TS]
y_TS = y_TS[index_TS]

# handling skewness
relevant_columns = ['DEP_DELAY', 'ARR_DELAY', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'DISTANCE']
for col in relevant_columns:
  X_TR[col] = 1 / (X_TR[col] + 1e-6)
  X_TS[col] = 1 / (X_TS[col] + 1e-6)

# handling outliers
for col in ['DEP_DELAY', 'DEP_DEL15', 'ARR_DELAY', 'CANCELLED', 'DIVERTED','ACTUAL_ELAPSED_TIME']:
        lower_threshold = X_TR[col].mean() - 3 * X_TR[col].std()
        upper_threshold = X_TR[col].mean() + 3 * X_TR[col].std()

        # Clip outliers in the training set
        X_TR[col] = np.clip(X_TR[col], lower_threshold, upper_threshold)

        # Clip outliers in the val set
        X_TS[col] = np.clip(X_TS[col], lower_threshold, upper_threshold)

# Handling class imbalance
sampler = SMOTE(sampling_strategy='auto', random_state=2)
X_TR, y_TR = sampler.fit_resample(X_TR, y_TR)

# List of classifiers
classifiers = {
          "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
          "Logistic Regression": LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=100),
          "Random Forest": RandomForestClassifier(random_state=0)
        }

results_df = pd.DataFrame(columns=['Method'] + list(classifiers.keys()))

for clf_name, clf in classifiers.items():
  # Initialize and train classifier
  clf.fit(X_TR, y_TR)

  # Predict on the test set
  y_pred = clf.predict(X_TS)

  # Calculate precision
  precision = round(precision_score(y_TS, y_pred), 4)
  precision_dict[clf_name] = precision

# Append precision results to the dataframe
precision_dict['Method'] = name
results_df = results_df.append(precision_dict, ignore_index=True)

# Sort the results dataframe by Method in ascending order
results_df = results_df.sort_values(by='Method', ascending=True)

# Print the results dataframe
print("FINAL RESULT ON TEST SET")
print(results_df)
