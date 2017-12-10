import numpy as np

def error_function(theta, X, y):
    '''
        Error function definition.
        theta,X,y - vectorized form.
        theta - parametrs.
    '''
    diff = np.dot(X, theta) - y
    return (1./(2*y.shape[0])) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''
        Gradient of the function.
        theta,X,y - vectorized form.
        theta - parametrs.      
    '''
    n = y.shape[0]
    i = np.random.randint(0, n)
    diff = np.dot(X[i,:], theta) - y[i]
    return (1./n) * np.dot(np.transpose(X[i,:])[:,np.newaxis], diff[:,np.newaxis])

def stochastic_gradient_descent(X, y, alpha = 0.001, num_iter = 10000):
    '''
        Perform stochastic gradient descent.
        X,y - vectorized form.
        alpha - the Learning Rate.
        num_iter - number of iterations.
    '''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    #while not np.all(np.absolute(gradient) <= 1e-5):
    for i in range(num_iter):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta