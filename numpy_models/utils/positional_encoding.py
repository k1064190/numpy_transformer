import numpy as np

def positional_encoding(max_position:int, d_model:int, min_freq:int=1e-4) -> np.array:
        
        ################## edit here ###################
    position = np.arange(max_position) # [max position]
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model) # [d_model]
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1) #[max_position , 1] * [1, d_model] = [ max position , d_model ]
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
  
    return pos_enc
    ################################################

class Embedding_with_positional_encoding_np:
    def __init__(self, num_emb, num_dim) -> None:
        self.num_emb = num_emb #vocab size
        self.num_dim = num_dim #embedding size
        
        self.forward_input = None
        
        limit = np.sqrt(2 / float(num_dim))
        self.W = np.random.normal(0.0, limit, size=(num_emb,num_dim))
        self.encoding_vector = positional_encoding(max_position=num_emb, d_model=num_dim)
        
    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array[int]): [# of batch, # of vocab(int) ] # [ [1, 3, 5, 10, 100 ...]  ]

        Returns:
            np.array: [# of batch, # of vocab, embedding_dim ] # [ [768] [768]                       ]
        """
        
        ################## edit here ###################
        output = self.W[x[:]] + self.encoding_vector[ :x.shape[1] ]
        self.forward_input = x
        #self.W =  [num_emb , num_dim]      
        #x = [# of batch, # of voab]
        
        return output
        ################################################
        
    def backward(self,d_prev:np.array) -> np.array:
        """

        self.W = [# of embedding , embedding_dim]
        
        Args:
            d_prev (np.array): [# of batch, # of vocab, embedding_dim]
            -> [# of batch, num_emb, embedding_dim]

        Returns:
            np.array: _description_
        """
        
        ################## edit here ###################
        
        b, vocab, dim = d_prev.shape
        vocab_len, dim = self.W.shape
        
        expanded_d_prev = np.zeros(shape=(b,vocab_len,dim))
        expanded_d_prev[:,self.forward_input[:]] = d_prev
        
        self.dW = np.sum(expanded_d_prev, axis=0) 

        return expanded_d_prev
        ################################################
    
    def __call__(self,x):
        return self.forward(x)

if __name__=="__main__":
    
    """
    model = Embedding_with_positional_encoding_np(10,20)
    x = np.random.randint(0,9, size=(1,5))
    output = model(x)
    model.backward(output)
    """
    
    import matplotlib.pyplot as plt
    ### Plotting ####
    d_model = 768
    max_pos = 256
    mat = positional_encoding(max_pos, d_model)
    plt.pcolormesh(mat, cmap='copper')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()
