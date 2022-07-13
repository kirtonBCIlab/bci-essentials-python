from xxlimited import new
import numpy as np
import math
import random
import matplotlib.pyplot as plt

X = [40.0, 11.1, 30.0, 21.4, 10.7, 3.4, 42.0, 31.1, 50.0, 60.4, 45.7, 17.3]
y = [36.0, 37.2, 36.5, 39.4, 39.6, 40.7, 37.6, 42.2, 38.5, 39.4, 38.6, 42.7]
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

def within_group_covariance(group0, group1):
        group0_covariance = np.cov(group0)
        group1_covariance = np.cov(group1)

        W_matrix = (group0_covariance + group1_covariance)/2

        return W_matrix

def between_group_covariance(complete_array, some_W_Matrix):
    total_variance = np.cov(complete_array)

    B_matrix = total_variance - some_W_Matrix

    return B_matrix

def compute_S_matrix(some_W_matrix, some_B_matrix):
    S_matrix = np.matmul(np.linalg.inv(some_W_matrix), some_B_matrix)
    return S_matrix

def find_eigenvalues(some_S_matrix):
    return np.linalg.eig(some_S_matrix)

def find_LD_scores(X, y, group0, group1, weight0, weight1):
    LD_scores = []

    for i in range(len(X)):
        score = X[i]*weight0 + y[i]*weight1
        LD_scores.append(score)

    group0_variance = np.cov(LD_scores[:int(len(LD_scores)/2)])
    group1_variance = np.cov(LD_scores[int(len(LD_scores)/2):])

    return group0_variance, group1_variance, LD_scores

def find_new_LD_scores(X, y, weight0, weight1):
    LD_scores_new = []

    for i in range(len(X)):
        score = X[i]*weight0 + y[i]*weight1
        LD_scores_new.append(score)

    return LD_scores_new

def find_threshold(scores_list):
    group0_scores = scores_list[:int(len(scores_list)/2)]
    group1_scores = scores_list[int(len(scores_list)/2):]

    group0_scores_mean = np.mean(group0_scores)
    group1_scores_mean = np.mean(group1_scores)

    final_mean = (group0_scores_mean + group1_scores_mean) / 2

    return group0_scores_mean, group1_scores_mean, final_mean

def predict(weight0, weight1, X, y, mean0, mean1, threshold_value):
    predicted_scores = []

    for i in range(len(X)):
        score = X[i]*weight0 + y[i]*weight1
        predicted_scores.append(score)

    if mean0 > threshold_value:
        predicted_classes = [0 if score > threshold_value else 1 for score in predicted_scores]
    else:
        predicted_classes = [1 if score > threshold_value else 0 for score in predicted_scores]

    return predicted_classes

def accuracy(y_pred, y_true):
        correct = 0

        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                correct += 1

        my_accuracy = correct/len(y_pred)
        return my_accuracy

both_groups_array = np.stack((X, y), axis=0)
group_0_array = np.stack((X[:int(len(X)/2)], y[:int(len(X)/2)]), axis=0)
group_1_array = np.stack((X[int(len(X)/2):], y[int(len(X)/2):]), axis=0)

W_matrix = within_group_covariance(group_0_array, group_1_array) # Correct

B_matrix = between_group_covariance(both_groups_array, W_matrix) # Correct

S_matrix = compute_S_matrix(W_matrix, B_matrix) # Correct

eigenvalues, eigenvalues_matrix = find_eigenvalues(S_matrix) # Correct

weight_a = eigenvalues_matrix[0, 0] # Correct
weight_c = eigenvalues_matrix[1, 0] # Correct

group0_var, group1_var, LD_scores_unweighted = find_LD_scores(X, y, group_0_array, group_1_array, weight_a, weight_c) # Correct

pooled_var = (group0_var + group1_var) / 2

new_weight_a = weight_a/math.sqrt(pooled_var) # Correct
new_weight_c = weight_c/math.sqrt(pooled_var) # Correct

LD_scores_new = find_new_LD_scores(X, y, new_weight_a, new_weight_c) # Correct

group0_mean, group1_mean, threshold = find_threshold(LD_scores_new)

predicted_classes = predict(new_weight_a, new_weight_c, X, y, group0_mean, group1_mean, threshold)

model_accuracy = accuracy(predicted_classes, labels)
print(model_accuracy)