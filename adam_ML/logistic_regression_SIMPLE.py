import random
import math
import matplotlib.pyplot as plt

def main():
    def create_dataset(len_dataset, user_input):

        if user_input == 'x':

            dataset = []

            for i in range(int(len_dataset/2)):
                temp_point = []

                temp_point.append(random.randint(-5, 0))
                temp_point.append(random.randint(0, 5))
                temp_point.append(0)

                dataset.append(temp_point)

            for i in range(int(len_dataset/2)):
                temp_point = []

                temp_point.append(random.randint(0, 5))
                temp_point.append(random.randint(0, 5))
                temp_point.append(1)

                dataset.append(temp_point)

            return dataset

        else:

            dataset = []

            for i in range(int(len_dataset/2)):
                temp_point = []

                temp_point.append(random.randint(-10, 10))
                temp_point.append(random.randint(0, 5))
                temp_point.append(1)

                dataset.append(temp_point)

            for i in range(int(len_dataset/2)):
                temp_point = []

                temp_point.append(random.randint(-10, 10))
                temp_point.append(random.randint(-5, 0))
                temp_point.append(0)

                dataset.append(temp_point)

            return dataset

    # Figure out what term lambda should be applied to
    def determine_lambda_weighting(dataset):
        class_0_sum_x = 0
        class_0_sum_y = 0

        class_1_sum_x = 0
        class_1_sum_y = 0

        for i in dataset:
            if i[2] == 0:
                class_0_sum_x += i[0]
                class_0_sum_y += i[1]
            else:
                class_1_sum_x += i[0]
                class_1_sum_y += i[1]

        class_0_mean_x = class_0_sum_x/len(dataset)
        class_0_mean_y = class_0_sum_y/len(dataset)

        class_1_mean_x = class_1_sum_x/len(dataset)
        class_1_mean_y = class_1_sum_y/len(dataset)

        x_diff = abs(class_1_mean_x - class_0_mean_x)
        y_diff = abs(class_1_mean_y - class_0_mean_y)

        if x_diff > y_diff:
            print("Theta 2 Weighted")
            return("t2")
        else:
            print("Theta 1 Weighted")
            return("t1")

    # Sigmoid function for classification
    def sigmoid(x):
        value = 1 / (1 + (math.e ** -x))
        return (value)

    # Seperate into X, y and labels
    def seperate_dataset(dataset):
        X = []
        y = []
        z = []

        for i in dataset:
            X.append(i[0])
            y.append(i[1])
            z.append(i[2])

        return (X, y, z)

    # Gradient Descent 
    def gradient_descent(lambda_value, weighting, lr, theta2, theta1, theta0, x, y, labels, steps):

        # find_derivatives() and gradient_descent() rely on one another
        def find_derivatives(lambda_value, weighting, lr, theta2, theta1, theta0, x, y, labels):
            theta0_derivative = 0
            theta1_derivative = 0
            theta2_derivative = 0

            y_predicted = []

            for i in range(len(y)):
                y_hat = theta0 + theta1*x[i] + theta2*y[i]
                sigmoid_y = sigmoid(y_hat)
                y_predicted.append(sigmoid_y)

            for i in range(len(x)):
                # Loss function/Cost function
                theta0_derivative += 1/len(x) * (y_predicted[i] - labels[i])
                if weighting == "t2":
                    theta1_derivative += 1/len(x) * theta1 * (y_predicted[i] - labels[i])
                    theta2_derivative += 1/len(x) * lambda_value * theta2 * (y_predicted[i] - labels[i])
                elif weighting == "t1":
                    theta1_derivative += 1/len(x) * lambda_value * theta1 * (y_predicted[i] - labels[i])
                    theta2_derivative += 1/len(x) * theta2 * (y_predicted[i] - labels[i])

            step_size_theta0 = theta0_derivative*lr
            new_theta0 = theta0 - step_size_theta0

            step_size_theta1 = theta1_derivative*lr
            new_theta1 = theta1 - step_size_theta1

            step_size_theta2 = theta2_derivative*lr
            new_theta2 = theta2 - step_size_theta2

            return(new_theta2, new_theta1, new_theta0, step_size_theta2, step_size_theta1, step_size_theta0)
        
        grade_theta2, grad_theta1, grad_theta0, step_theta2, step_theta1, step_theta0 = find_derivatives(lambda_value, weighting, lr, theta2, theta1, theta0, x, y, labels)
        iterations = 0

        # Iterate until value is converged on
        while ((abs(step_theta0) > 0.000001 or abs(step_theta1 > 0.000001 or abs(step_theta2)) > 0.000001) and iterations < steps):
            if iterations == 0:
                theta2, theta1, theta0, step_theta2, step_theta1, step_theta0 = find_derivatives(lambda_value, weighting, lr, grade_theta2, grad_theta1, grad_theta0, x, y, labels)
            else :
                theta2, theta1, theta0, step_theta2, step_theta1, step_theta0 = find_derivatives(lambda_value, weighting, lr, theta2, theta1, theta0, x, y, labels)
            iterations += 1

        return (theta2, theta1, theta0, iterations)

    # Predict values passed in using found parameters
    def predict(theta0, theta1, theta2, x, y):
        y_hat = []
        for i in range(len(x)):
            y_value = sigmoid(theta0 + (theta1*x[i]) + (theta2*y[i]))
            y_hat.append(y_value)
        
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_hat]

        return y_predicted_classes

    # Finding accuracy of classifier
    # Finding accuracy on trained data (not best but shouldn't make too much of a difference)
    def accuracy(y_pred, y_true):
        correct = 0

        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                correct += 1

        my_accuracy = correct/len(y_pred)
        return my_accuracy

    my_input = input("Do you want x-weighted or y-weighted?: ")

    my_dataset = create_dataset(50, my_input)
    X, y, z = seperate_dataset(my_dataset)
    my_weighting = determine_lambda_weighting(my_dataset)

    theta0_intial = 1 # Bias term
    theta1_initial = 1
    theta2_initial = 1

    learning_rate = 0.01 # alpha = 0.1 is better but it throws an error
    iterations = 10000000
    my_lambda = 10

    final_theta2, final_theta1, final_theta0, iterations  = gradient_descent(my_lambda, my_weighting, learning_rate, theta0_intial, theta1_initial, theta2_initial, X, y, z, 1000000)
    print(f"Intercept: {final_theta0}")
    print(f"Weight 1: {final_theta1}")
    print(f"Weight 2: {final_theta2}")
    print(f"Iterations: {iterations}\n")

    predicted_classes = predict(final_theta0, final_theta1, final_theta2, X, y)
    final_accuracy = accuracy(predicted_classes, z)
    print(final_accuracy)

if __name__ == "__main__":
    main()