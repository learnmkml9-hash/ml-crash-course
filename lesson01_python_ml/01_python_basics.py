# Lesson 1, Part 1: Python basics for ML coding

# -----------------------------
# 1. Variables
# -----------------------------

learning_rate = 0.1
num_iterations = 100
model_name = "linear regression"

print("Model:", model_name)
print("Learning rate:", learning_rate)
print("Number of iterations:", num_iterations)


# -----------------------------
# 2. Lists
# -----------------------------

loss_values = [10.0, 6.5, 4.2, 2.8, 1.9]

print("\nLoss values:")
print(loss_values)

print("First loss:", loss_values[0])
print("Last loss:", loss_values[-1])


# -----------------------------
# 3. Loops
# -----------------------------

print("\nPrinting loss values one by one:")

for loss in loss_values:
    print(loss)


print("\nPrinting iteration number and loss:")

for i, loss in enumerate(loss_values):
    print("Iteration:", i, "Loss:", loss)


# -----------------------------
# 4. Dictionaries
# -----------------------------

experiment_config = {
    "algorithm": "gradient descent",
    "learning_rate": 0.1,
    "num_iterations": 100,
    "regularization": None,
}

print("\nExperiment config:")
print(experiment_config)

print("Algorithm:", experiment_config["algorithm"])
print("Learning rate:", experiment_config["learning_rate"])


# -----------------------------
# 5. Functions
# -----------------------------

def squared_error(y_true, y_pred):
    error = y_true - y_pred
    return error ** 2


example_loss = squared_error(y_true=3.0, y_pred=2.5)

print("\nExample squared error:")
print(example_loss)


# -----------------------------
# 6. A simple ML-style function
# -----------------------------

def predict_linear_model(x, weight, bias):
    y_pred = weight * x + bias
    return y_pred


x = 2.0
weight = 3.0
bias = 1.0

prediction = predict_linear_model(x, weight, bias)

print("\nLinear model prediction:")
print("x =", x)
print("weight =", weight)
print("bias =", bias)
print("prediction =", prediction)