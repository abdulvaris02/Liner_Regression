import numpy as np
import matplotlib.pyplot as plt

theta = np.array([1, 2])
def h(x, theta):
    return np.dot(x, theta)
  
  y_predicted = h(x, theta)
  
def mean_squared_error(y_predicted, y_label):
    return np.sum((y_predicted - y_label)**2)/len(y_label)

class LeastSquaresRegression():
    def __init__(self,):
        self.theta_ = None
        
    def fit(self, X, y):
        # Calculates theta that minimizes the MSE and updates self.theta_
        #θ = (XT·X)-1·XT·y
        #t_1 = np.linalg.inv(np.dot(X.T, X))
        #t_2 = np.dot(X.T, y)
            self.theta_ = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    def predict(self, X):
        # Make predictions for data X, i.e output y = h(X) (See equation in Introduction)
        return h(X, self.theta_)

X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)
plt.scatter(X, y)

X = 4 * np.random.rand(100, 1)

def bias_column(X):
    return np.append(np.ones((len(X),1)), X, axis = 1)

X_new = bias_column(X)
print(X[:5])
print(" ---- ")
print(X_new[:5])

model = LeastSquaresRegression()
model.fit(X_new, y)
print(model.theta_)

y_new = model.predict(X_new)

def my_plot(X, y, y_new):
    plt.scatter(X, y)
    plt.plot(X_new[:, 1], y_new, 'r')
    plt.grid()
    my_plot(X, y, y_new)
    
class GradientDescentOptimizer():
    def __init__(self, f, fprime, start, learning_rate = 0.1):
        self.f_      = f                       # The function
        self.fprime_ = fprime                  # The gradient of f
        self.current_ = start                  # The current point being evaluated
        self.learning_rate_ = learning_rate    # Does this need a comment ?
        # Save history as attributes
        self.history_ = start
    def step(self):
        # Take a gradient descent step
        # 1. Compute the new value and update selt.current_
        self.current_ = self.current_ - self.learning_rate_*fprime(self.current_)
        # 2. Append the new value to history
        self.history_ = np.append(self.history_, self.current_, axis = 1)
        # Does not return anything
        #return  self.current_
    def optimize(self, iterations = 100):
        # Use the gradient descent to get closer to the minimum:
        # For each iteration, take a gradient step
        for i in range(iterations):
            self.step()
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))
        return self.history_
      
    def plot_it(self):
        plt.plot(self.history_[0,:])
        plt.plot(self.history_[1,:])    
                
def f(x):
    a = np.array([[2],[6]])
    return 3 + np.dot((x-a).T, (x-a))
  
def fprime(x):
    a = np.array([[2],[6]])
    return 2 * (x-a)
  
grad = GradientDescentOptimizer(f, fprime, np.random.normal(size=(2,1)), 0.1)
grad.optimize(100)
grad.step()
grad.plot_it()

