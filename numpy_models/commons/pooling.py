import numpy as np

class MaxPooling2D:

    def __init__(self, pooling_shape: tuple, stride: int=None) -> None:
        self.pool_w, self.pool_h = pooling_shape
        self.stride = stride if stride!=None else self.pool_h
        
        self.d_zeros = None
    
    def forward(self,x):
        self.d_zeros = np.zeros_like(x)
        ######################   edit here   ##############################
        # x.shape = [# of batch, # of channel, input_height, input_width]
        # output.shape = [# of batch, # of channel, output_height, output_widh]
        
        n, c, in_h, in_w = x.shape
        h_out = (in_h - self.pool_h) // self.stride + 1
        w_out = (in_w - self.pool_w) // self.stride + 1
        
        output = np.zeros(shape=(n,c,h_out,w_out))
        self.d_zeros = np.zeros(shape=(n,c,in_h,in_w)) # 0 1
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w

                sub_array = x[:,:, h_start:h_end, w_start:w_end]
                
                output[:,:, i:i+1 , j:j+1] = np.max( a =  sub_array, axis=(2,3)).reshape(n,c,1,1)
                self.d_zeros[:,:, h_start:h_end , w_start:w_end] = self.save_max_idx(sub_array)

        self.output = output        
        return output
        
            
    def save_max_idx(self, a:np.array) -> np.array:
        # [# of batch, # of channel, pooling_height, pooling_width]
        n, c, h, w = a.shape
        a = a.reshape( n,c, h*w )
        idx = np.argmax(a,axis=2)
        
        output = np.zeros_like(a)
        output[np.arange(n)[:, np.newaxis, np.newaxis], np.arange(c)[np.newaxis, :, np.newaxis], idx[:, :, np.newaxis]] = 1
        output = output.reshape(n,c,h,w)
        

        return output
        ###################################################################
    
    def backward(self,d_prev):
        ######################   edit here   ##############################
        # output = [# of batch, # of channel, input_height, input_width]
        # input = [# of batch, # of channel, output_height, output_widh]    
        n, c, in_h, in_w = self.d_zeros.shape
        h_out = (in_h - self.pool_h) // self.stride + 1
        w_out = (in_w - self.pool_w) // self.stride + 1
        
        output = np.zeros(shape=(n,c,in_h,in_w))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w
                
                output[:,:,h_start:h_end, w_start:w_end] = d_prev[:,:,i:i+1, j:j+1] * self.d_zeros[:,:,h_start:h_end, w_start:w_end]
            
        return output
        ###################################################################