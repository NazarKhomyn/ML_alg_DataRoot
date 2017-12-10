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
    diff = np.dot(X, theta) - y
    return (1./y.shape[0]) * np.dot(np.transpose(X), diff)


def batch_gradient_descent(X, y, alpha):
    '''
        Perform batch gradient descent.
        X,y - vectorized form.
        alpha - the Learning Rate.
    '''    
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-3):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta