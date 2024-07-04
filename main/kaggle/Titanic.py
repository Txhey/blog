# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the training data
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Drop the 'Cabin' column since it has too many missing values
train_data.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)
test_data.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

# # Plot the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
# plt.show()

# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.show()

# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()

# Create a new feature 'FamilySize'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Encode categorical features using get_dummies
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])

# Align the train and test data
X = train_data.drop(columns=['Survived', 'PassengerId'])
y = train_data['Survived']
X_test = test_data.drop(columns=['PassengerId'])

# Ensure the test data has the same dummy variable columns as the train data
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Parameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_val)
print(f'Best Model Validation Accuracy: {accuracy_score(y_val, y_pred_best)}')

# Predict on the test data
test_pred = best_model.predict(X_test)

# Create a submission DataFrame and save to a CSV file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
