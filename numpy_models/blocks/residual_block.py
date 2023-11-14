from typing import Any
import numpy as np

class Residual_block_np:
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self,x:np.array):
        ############### edit here ###############
        pass
        #########################################
    
    def backward(self,d_prev):
        ############### edit here ###############
        pass
        #########################################
    
    def __call__(self,x):
        return self.forward(x)