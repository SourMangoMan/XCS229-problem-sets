import numpy as np

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        y = np.array(y)

        n = x.shape[0]
        no_of1s = 0
        no_of0s = 0
        mu0_top = 0
        mu1_top = 0
        sigma = np.zeros((x.shape[1], x.shape[1]))

        for i in range(n):
            no_of1s += y[i]
            no_of0s += 1 - y[i]
            mu0_top += (1 - y[i]) * x[i, :]
            mu1_top += (y[i]) * x[i, :]
            
        phi = no_of1s / n
        mu_0 = np.array(mu0_top / no_of0s).reshape((-1, 1))
        
        mu_1 = np.array(mu1_top / no_of1s).reshape((-1, 1))
        

        for i in range(n):
            x_i = (x[i, :]).reshape((-1, 1))

            if y[i] < 0.5:
                sigma += (x_i - mu_0) @ (x_i - mu_0).T 
            else:
                sigma += (x_i - mu_1) @ (x_i - mu_1).T 

        sigma /= n

        # You need to calculate self.theta!!!!!!! 
        S = np.linalg.inv(sigma)
        theta = (S @ (mu_1 - mu_0))
        theta_0 = 0.5 * (mu_0.T @ S @ mu_0 - mu_1.T @ S @ mu_1) - np.log((1 - phi) / phi)

        if theta_0.shape != (1, 1):
            theta_0 = theta_0.reshape((1, 1))

        self.theta = np.vstack((theta_0, theta))
        # self.theta = 

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        
        
        linear = (self.theta.T @ x.T).T
        prediction = 1 / (1 + np.exp(-linear))
        prediction = prediction.reshape(x.shape[0])

        return (prediction >= 0.5).astype(int)
        # *** END CODE HERE