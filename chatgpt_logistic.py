import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = np.array([[22, 18000, 0],
                 [25, 22000, 0],
                 [47, 50000, 1],
                 [52, 60000, 1],
                 [46, 70000, 1],
                 [56, 80000, 1],
                 [55, 120000, 1],
                 [60, 140000, 1],
                 [62, 150000, 1],
                 [23, 18000, 0],
                 [35, 22000, 0]])
# Separate features (X) and target variable (y)
X = data[:, :-1]  # Features: age, income
y = data[:, -1]   # Target variable: bought_insurance

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
