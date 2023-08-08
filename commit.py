import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Convert GDP growth to binary labels: 1 for positive growth and 0 for negative growth
y_train_binary = np.where(y_train >= 0, 1, 0)
y_test_binary = np.where(y_test >= 0, 1, 0)

# Check if both classes (0 and 1) are present in the binary labels
if np.unique(y_train_binary).size == 2 and np.unique(y_test_binary).size == 2:
    # Create a Logistic Regression model
    logistic_model = LogisticRegression()

    # Train the model using the training data
    logistic_model.fit(X_train, y_train_binary)

    # Predict the binary labels on the testing data
    y_pred_binary = logistic_model.predict(X_test)

    # Evaluate the model's performance using accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
    class_report = classification_report(y_test_binary, y_pred_binary)

    print("Logistic Regression Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
else:
    print("Binary classification is not possible as the data contains samples from only one class.")
