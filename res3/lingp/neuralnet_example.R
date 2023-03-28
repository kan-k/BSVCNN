# First, install and load the neuralnet package
# install.packages("neuralnet")
library(neuralnet)
library(MASS)

# Next, prepare your data. This example uses the Boston  dataset.
data(Boston)

# # Split the data into training and testing sets
# train_indices <- sample(1:nrow(Boston), 0.8*nrow(Boston))
# train_data <- Boston[train_indices, -1]
# train_labels <- Boston[train_indices, 1]
# test_data <- Boston[-train_indices, -1]
# test_labels <- Boston[-train_indices, 1]
# 
# # Build the neural network model using the neuralnet() function
# # and set the error function as "error" and activation function as "linear"
# model <- neuralnet(medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + b + lstat, 
#                    data = train_data, hidden = 1, error.fct = "sse", linear.output = TRUE)
# 
# # Predict the output for the test data using the compute() function
# predictions <- compute(model, test_data)$net.result
# 
# # Compare predictions to true labels to evaluate the accuracy of the model
# rmse <- sqrt(mean((test_labels - predictions)^2))


# Split the data into training and testing sets
train_indices <- sample(1:nrow(Boston), 0.8*nrow(Boston))
train_data <- Boston[train_indices,] 
# train_labels <- Boston[train_indices, 1]
test_data <- Boston[-train_indices,]
test_labels <- Boston[-train_indices, 14]

train_data2 <-  Boston[train_indices,1:4]
train_labels <- Boston[train_indices, 14]


# Build the neural network model using the neuralnet() function
# and set the error function as "error" and activation function as "linear"
model <- neuralnet(medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, 
                    data = train_data, hidden = 13, linear.output = TRUE)

model2 <- neuralnet( train_labels ~., hidden = 5, err.fct = 'sse', linear.output = TRUE, data = train_data2)

# Predict the output for the test data using the compute() function
predictions <- compute(model, test_data)$net.result
predictions2 <- compute(model2, test_data)$net.result

# Compare predictions to true labels to evaluate the accuracy of the model
(rmse1 <- sqrt(mean((test_labels - predictions)^2)))
(rmse2 <- sqrt(mean((test_labels - predictions2)^2)))

plot(model2, x.text = test_labels, y.text = predictions2, errors=rmse2)

# Plot the error trajectory
plot(model$stats$error, type = "l", xlab = "Iterations", ylab = "Error")
# 












