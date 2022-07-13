import random
import matplotlib.pyplot as plt

def main():
    # Create a random dataset
    def create_dataset(len_dataset):
        X = []
        Y = []
        
        for i in range(len_dataset):
            X.append(i)
            Y.append(random.randint(i, i+10))
            
        return (X,Y)

    # Gradient Descent 
    def gradient_descent(lr, lf, slope, intercept, x, y, steps):

        # find_derivatives() and gradient_descent() rely on one another
        def find_derivatives(lr, lf, slope, intercept, x, y):
            i_derivative = 0
            s_derivative = 0

            if lf == "SSR":
                for i in range(len(x)):
                    # Loss function/Cost function
                    i_derivative += -2*(y[i] - (intercept + (slope*x[i])))
                    s_derivative += (-2*x[i])*(y[i] - (intercept + (slope*x[i])))

            step_size_int = i_derivative*lr
            new_int = intercept - step_size_int

            step_size_slope = s_derivative*lr
            new_slope = slope - step_size_slope

            return(new_slope, new_int, step_size_slope, step_size_int)
        
        grad_m, grad_b, step_s, step_i = find_derivatives(lr, lf, slope, intercept, x, y)
        iterations = 0

        # Iterate until value is converged on
        while ((abs(step_i) > 0.00000001 or abs(step_s) > 0.00000001) and iterations < steps):
            if iterations == 0:
                m, b, step_s, step_i = find_derivatives(lr, lf, grad_m, grad_b, x, y)
            else :
                m, b, step_s, step_i = find_derivatives(lr, lf, m, b, x, y)
            iterations += 1

        return (m, b, iterations)

    # Creating trendline for plotting
    def create_trendline(slope, intercept, x):
        trend_y = []
        for i in range(len(x)):
            trend_y.append(slope*x[i]+intercept)
            
        return trend_y

    learning_rate = 0.001
    # Sum of squared residuals
    loss_function = "SSR"

    my_x, my_y = create_dataset(10)

    # Starting values: m = 0 and b = 0
    final_slope, final_intercept, iterations = gradient_descent(learning_rate, loss_function, 0, 0, my_x, my_y, 10000000)
    print("Predicted slope is: %s" % (final_slope))
    print("Predicted int is: %s" % (final_intercept))

    print("Number of interations: %s" % (iterations))

    trendline  = create_trendline(final_slope, final_intercept, my_x)

    # Plotting
    plt.scatter(my_x,my_y)
    plt.plot(my_x, trendline, color="orange")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Linear Regression example")
    plt.show()

if __name__ == "__main__":
    main()
