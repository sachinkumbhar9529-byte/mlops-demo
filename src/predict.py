import joblib

# Load the trained model
model = joblib.load("../models/iris_model.pkl")

# Define sample flower measurements
sample_data = [[5.1, 3.5, 1.4, 0.2]]

# Make a prediction
prediction = model.predict(sample_data)

# Display the prediction
print("Predicted class:", prediction)