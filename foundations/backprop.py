import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, X: NDArray[np.float64], W: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        x = np.array(X)
        w = np.array(W)
        y_true = np.array(y_true)
        z = np.dot(x,w) + b
        y_hat = 1/(1+ np.exp(-z)) 
        loss = 0.5 * np.square(y_hat - y_true)
        # calculate the error term
        # Corrected error term
        error_term = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_db = np.mean(error_term)
        dL_dw = np.dot(error_term, x)
        return (np.round(dL_dw, 5), round(float(dL_db), 5))


        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
