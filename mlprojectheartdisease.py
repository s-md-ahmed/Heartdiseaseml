import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
np.random.seed(42)
# Load the dataset
data1 = pd.read_csv("D:/MLproject datasets/heart_disease_uci.csv")
print(data1.head())

# Drop unnecessary columns
data1.drop(['id', 'dataset'], axis=1, inplace=True)
data1.info()

# Display descriptive statistics
data1.describe()

# Separate numeric and categorical variables for visualization purposes
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

heart_cat = data1[CATEGORICAL_COLS]
heart_num = data1[NUMERICAL_COLS]

heart_cat.nunique()

# Visualize the distribution of categorical variables
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Countplots for categorical variables
sns.countplot(x='sex', data=heart_cat, ax=axes[0, 0])
axes[0, 0].set_title('Gender Distribution')
sns.countplot(x='cp', data=heart_cat, ax=axes[0, 1])
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].set_title('Chest Pain Types')
sns.countplot(x='fbs', data=heart_cat, ax=axes[0, 2])
axes[0, 2].set_title('Fasting Blood Sugar > 120 mg/dl')
sns.countplot(x='restecg', data=heart_cat, ax=axes[0, 3])
axes[0, 3].set_title('Resting Electrocardiographic Results')
sns.countplot(x='exang', data=heart_cat, ax=axes[1, 0])
axes[1, 0].set_title('Exercise Induced Angina')
sns.countplot(x='slope', data=heart_cat, ax=axes[1, 1])
axes[1, 1].set_title('Slope of the Peak Exercise ST Segment')
sns.countplot(x='thal', data=heart_cat, ax=axes[1, 2])
axes[1, 2].set_title('Defects')
sns.countplot(x='ca', data=heart_cat, ax=axes[1, 3])
axes[1, 3].set_title('Number of Major Vessels colored by Fluoroscopy')
plt.tight_layout()
plt.show()

# Use scatterplots to visualize key relationships in numerical data
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Scatterplots for numerical variables
heart_num.plot('age', 'chol', kind='scatter', ax=axes[0, 0])
axes[0, 0].set_title('Age Against Cholesterol Levels')
heart_num.plot('age', 'trestbps', kind='scatter', ax=axes[0, 1])
axes[0, 1].set_title('Age Against Resting Blood Pressure')
heart_num.plot('age', 'thalch', kind='scatter', ax=axes[1, 0])
axes[1, 0].set_title('Age Against Maximum Heart Rate Achieved')
heart_num.plot('age', 'oldpeak', kind='scatter', ax=axes[1, 1])
axes[1, 1].set_title('Age Against ST Depression')
plt.tight_layout()
plt.show()

# Visualize relationships using scatterplots
fig, axes = plt.subplots(3, figsize=(7, 10))

sns.scatterplot(x='chol', y='thalch', hue='num', data=data1, ax=axes[0])
axes[0].set_title('Effect of Cholesterol on Maximum Heart Rate')
sns.scatterplot(x='chol', y='thalch', hue='sex', data=data1, ax=axes[1])
sns.scatterplot(x='chol', y='thalch', hue='restecg', data=data1, ax=axes[2])
plt.show()

# Display statistics based on the target variable
data1.groupby('num').mean()
print('Average Cholesterol Level Based on Target Variable and Chest Pain Type')
# Display cross-tabulation
print(pd.crosstab(index=data1.num, columns=data1.cp, values=data1.chol, aggfunc=np.mean))

# Display correlation matrix and heatmap
corr = data1.corr()
print(corr)

sns.heatmap(corr)
plt.show()

# Display boxplot to visualize outliers in the data
data1.boxplot()
plt.show()

# Handling missing values and outliers
data1.loc[data1['chol'] == 0, :]
data1.info()
data1['cp'].isnull().sum()
data1['slope'].isnull().sum()

# Cholesterol Levels
median_chol = data1.loc[data1['chol'] != 0, 'chol'].median()
heart_df = data1.fillna(value={'chol': median_chol})
heart_df.loc[heart_df['chol'] == 0, 'chol'] = median_chol

# Resting Blood Pressure
mean_bp = heart_df.loc[heart_df['trestbps'] != 0, 'trestbps'].mean()
heart_df = heart_df.fillna(value={'trestbps': mean_bp})
heart_df.loc[heart_df['trestbps'] == 0, 'trestbps'] = mean_bp

# Maximum Heart Rate
mean_hr = heart_df.loc[heart_df['thalch'] != 0, 'thalch'].mean()
heart_df = heart_df.fillna(value={'thalch': mean_hr})
heart_df.loc[heart_df['thalch'] == 0, 'thalch'] = mean_hr

# Old Peak
mean_peak = heart_df.oldpeak.mean()
heart_df = heart_df.fillna(value={'oldpeak': mean_peak})
heart_df.loc[heart_df['oldpeak'] == 0, 'oldpeak'] = mean_peak
# Apply RobustScaler for handling outliers
robust_scaler = RobustScaler()
heart_df[NUMERICAL_COLS] = robust_scaler.fit_transform(heart_df[NUMERICAL_COLS])
# Feature scaling using Min-Max scaling
scaler = MinMaxScaler()
heart_df[NUMERICAL_COLS] = scaler.fit_transform(heart_df[NUMERICAL_COLS])

# Drop columns with a great number of missing values and reassign datatypes
heart_df.drop(labels=['ca', 'thal', 'slope'], axis=1, inplace=True)
heart_df = heart_df.astype({'sex': 'category', 'cp': 'category', 'fbs': 'bool', 'restecg': 'category', 'exang': 'bool'})

# Drop remaining rows with missing values and display distribution for target variables
heart_df.dropna(inplace=True)
sns.countplot(x='num', data=heart_df)
plt.show()

# Splitting the data into training and testing sets
heart_onehot = pd.get_dummies(heart_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang'])
print(heart_onehot.head())
X = heart_onehot.drop('num', axis=1)
y = heart_onehot.num
# Using SMOTE for dealing with imbalanced classes before splitting
smt = SMOTE(sampling_strategy='not majority')
X_resampled, y_resampled = smt.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Check class distribution in the target variable
y_train.value_counts()


# DecisionTreeClassifier after applying SMOTE
clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

# RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=11,max_depth=11)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# GradientBoostingClassifier
gradient_booster = GradientBoostingClassifier(learning_rate=0.02, max_depth=3, n_estimators=150)
gradient_booster.fit(X_train, y_train)
y_pred_gb = gradient_booster.predict(X_test)

# AdaBoostClassifier
ada_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)

# Display classification reports for all classifiers
# Random Forest
print("Random Forest Training Set Classification Report:")
print(classification_report(y_train, clf_rf.predict(X_train), zero_division=1))

print("\nRandom Forest Testing Set Classification Report:")
print(classification_report(y_test, clf_rf.predict(X_test), zero_division=1))

# Naive Bayes
print("\nNaive Bayes Training Set Classification Report:")
print(classification_report(y_train, nb_classifier.predict(X_train), zero_division=1))

print("\nNaive Bayes Testing Set Classification Report:")
print(classification_report(y_test, nb_classifier.predict(X_test), zero_division=1))

# SVM
print("\nSupport Vector Machine (SVM) Training Set Classification Report:")
print(classification_report(y_train, svm_classifier.predict(X_train), zero_division=1))

print("\nSupport Vector Machine (SVM) Testing Set Classification Report:")
print(classification_report(y_test, svm_classifier.predict(X_test), zero_division=1))

# Gradient Boosting
print("\nGradient Boosting Training Set Classification Report:")
print(classification_report(y_train, gradient_booster.predict(X_train), zero_division=1))

print("\nGradient Boosting Testing Set Classification Report:")
print(classification_report(y_test, gradient_booster.predict(X_test), zero_division=1))

# AdaBoost
print("\nAdaBoost Training Set Classification Report:")
print(classification_report(y_train, ada_classifier.predict(X_train), zero_division=1))

print("\nAdaBoost Testing Set Classification Report:")
print(classification_report(y_test, ada_classifier.predict(X_test), zero_division=1))

# Compare F1 scores with a bar chart
classifiers = ['Random Forest', 'Naive Bayes', 'SVM', 'Gradient Boosting', 'AdaBoost']
reports = [classification_report(y_test, y_pred_rf, output_dict=True),
           classification_report(y_test, y_pred_nb, output_dict=True),
           classification_report(y_test, y_pred_svm, output_dict=True),
           classification_report(y_test, y_pred_gb, output_dict=True),
           classification_report(y_test, y_pred_ada, output_dict=True)]

# Extract precision, recall, and F1 score for each classifier
precision_scores = [report['macro avg']['precision'] for report in reports]
recall_scores = [report['macro avg']['recall'] for report in reports]
f1_scores = [report['macro avg']['f1-score'] for report in reports]

# Create a horizontal bar plot
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Precision plot
sns.barplot(x=precision_scores, y=classifiers, palette='viridis', ax=axes[0])
axes[0].set_title('Comparison of Precision Scores for Different Classifiers')
axes[0].set_xlabel('Precision Score')
axes[0].set_ylabel('Classifier')

# Recall plot
sns.barplot(x=recall_scores, y=classifiers, palette='viridis', ax=axes[1])
axes[1].set_title('Comparison of Recall Scores for Different Classifiers')
axes[1].set_xlabel('Recall Score')
axes[1].set_ylabel('Classifier')

# F1 score plot
sns.barplot(x=f1_scores, y=classifiers, palette='viridis', ax=axes[2])
axes[2].set_title('Comparison of F1 Scores for Different Classifiers')
axes[2].set_xlabel('F1 Score')
axes[2].set_ylabel('Classifier')

plt.tight_layout()
plt.show()
import pickle

# Save RandomForestClassifier
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(clf_rf, file)

# Save MinMaxScaler
with open('min_max_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
# Save RobustScaler
with open('robust_scaler.pkl', 'wb') as file:
    pickle.dump(robust_scaler, file)
# User input for predictions
user_input = []

print("\nEnter the following details for prediction:")
for col in X.columns:
    val = input(f"{col}: ")
    user_input.append(float(val))

# Convert the user input into a DataFrame
user_df = pd.DataFrame([user_input], columns=X.columns)

# Make predictions for each classifier
predictions = {
    'Random Forest': clf_rf.predict(user_df),
    #'Naive Bayes': nb_classifier.predict(user_df),
    #'SVM': svm_classifier.predict(user_df),
    #'Gradient Boosting': gradient_booster.predict(user_df),
    #'AdaBoost': ada_classifier.predict(user_df)
}

# Display predictions
print("\nPredictions:")
for classifier, prediction in predictions.items():
    print(f"{classifier}: {prediction[0]}")