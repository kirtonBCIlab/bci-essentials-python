# Least squares method which will return most accurate value
def find_starting_values(x, y):
    x_sum = 0
    y_sum = 0
    xy = 0
    x_sqaured = 0

    for i in x:
        x_sum += i
        x_sqaured += i**2
    for i in y:
        y_sum += i
    
    for i in range(len(x)):
        xy += (x[i] * y[i])

    slope = (((len(x)*xy) - (x_sum*y_sum)) / (len(x)*x_sqaured - (x_sum**2)))
    intercept = ((y_sum - (slope*x_sum)) / len(x))

    return (slope, intercept)