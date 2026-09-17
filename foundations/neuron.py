import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, X: NDArray[np.float64], W: NDArray[np.float64], b: float, activation: str) -> float:
        x = np.array(X)
        w = np.array(W) 
        z = np.dot(x,w) + b
        if activation == "sigmoid":
            result = 1 / (1+ np.exp(-z))
            return round(result, 5)
        elif activation == "relu":
            result = max(0.0,z)
            return round(result, 5)
        else:
            raise valueerror("Please select either 'ReLU' or 'sigmoid' as an activation fuction ")





        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
