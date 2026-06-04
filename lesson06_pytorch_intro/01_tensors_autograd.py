# Lesson 6, Part 1: PyTorch Tensors and Autograd

import torch


# ------------------------------------------------------------
# 1. Basic tensors
# ------------------------------------------------------------

x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.5, -1.0, 2.0])
b = torch.tensor(0.1)

print("x:", x)
print("w:", w)
print("b:", b)

print("\nShapes:")
print("x shape:", x.shape)
print("w shape:", w.shape)
print("b shape:", b.shape)


# ------------------------------------------------------------
# 2. Tensor operations
# ------------------------------------------------------------

dot_product = torch.dot(w, x)
prediction = dot_product + b

print("\nDot product:")
print(dot_product)

print("\nPrediction:")
print(prediction)


# ------------------------------------------------------------
# 3. Autograd: scalar example
# ------------------------------------------------------------

a = torch.tensor(3.0, requires_grad=True)

loss = a ** 2 + 2 * a + 1

print("\nScalar autograd example:")
print("a:", a)
print("loss:", loss)

loss.backward()

print("d loss / d a:")
print(a.grad)


# ------------------------------------------------------------
# 4. Autograd: linear regression one example
# ------------------------------------------------------------

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor(10.0)

w = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = torch.dot(w, x) + b
loss = (y_pred - y_true) ** 2

print("\nLinear regression one-example autograd:")
print("y_pred:", y_pred)
print("loss:", loss)

loss.backward()

print("\nGradients:")
print("w.grad:", w.grad)
print("b.grad:", b.grad)


# ------------------------------------------------------------
# 5. Important: gradients accumulate
# ------------------------------------------------------------

# If we call backward again without clearing gradients,
# PyTorch adds new gradients to old gradients.
# This is why training loops use optimizer.zero_grad().

y_pred_2 = torch.dot(w, x) + b
loss_2 = (y_pred_2 - y_true) ** 2
loss_2.backward()

print("\nGradients after calling backward a second time:")
print("w.grad:", w.grad)
print("b.grad:", b.grad)

print("\nKey lesson: PyTorch accumulates gradients unless we clear them.")