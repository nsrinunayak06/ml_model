# train_knn_model.py
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Example: Use Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save the model to file
joblib.dump(model, 'knn_model.pkl')
