import numpy as np

class Flatten_np:
    def __init__(self) -> None:
        self.origin_shape = None
    
    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array): example) [# of batch, a, b, c]

        Returns:
            np.array: [# of batch, a*b*c ]
            
        """
        
        self.origin_shape = np.shape(x)
        x = x.reshape(self.origin_shape[0], -1)
        
        return x
    
    def backward(self,d_prev:np.array) -> np.array:
        """

        Args:
            x (np.array): example) [# of batch, a*b*c]

        Returns:
            np.array: [# of batch, a, b, c ]
        """

        d_prev = d_prev.reshape(self.origin_shape)
        
        return d_prev
    
    def __call__(self,x):
        return self.forward(x)


if __name__=="__main__":
    flatten_layer = Flatten_np()
    x = np.random.rand(2,5,10)
    out = flatten_layer(x)
    x_prev = flatten_layer.backward(out)
    print(x.shape)
    print(out.shape)
    print(x_prev.shape)