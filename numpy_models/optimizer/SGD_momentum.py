import numpy as np

class SGD_momentum_np:
    def __init__(self, alpha: float = 0.9) -> None:
        self.velocity = dict()
        self.alpha = alpha
        """
        alpha: default 0.9
        
        W(t+1) = W(t) + lr * V(t)
        
        V(t) = a * V(t-1) - cost
        
        """
    def update_grad(self, layer_name:str, layer, LR:float, have_db:bool):
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
            have_db (bool): layer에 dW외에 db가 있는지 유무, default=True
        """
        ################## edit here ###################
        self.save_velocity(layer_name,layer,have_db)

        layer.W = layer.W + self.velocity[f"{layer_name}_W"] * LR

        if not have_db:
            return layer

        layer.b = layer.b + self.velocity[f"{layer_name}_b"] * LR

        return layer
        ################################################

    def save_velocity(self,layer_name, layer, have_db):
        
        ################## edit here ###################
        
        if f"{layer_name}_W" not in self.velocity.keys():
            self.velocity[f"{layer_name}_W"] = 0
        else:
            self.velocity[f"{layer_name}_W"] = self.alpha * self.velocity[f"{layer_name}_W"] - layer.dW 
        
        if not have_db:
            return
        
        if f"{layer_name}_b" not in self.velocity.keys():
            self.velocity[f"{layer_name}_b"] = 0
        else:
            self.velocity[f"{layer_name}_b"] = self.alpha * self.velocity[f"{layer_name}_b"] - layer.db 

        ################################################