import numpy as np

class Embedding_np:
    def __init__(self, num_emb, num_dim) -> None:
        self.num_emb = num_emb #vocab size
        self.num_dim = num_dim #embedding size
        
        self.forward_input = None
        
        limit = np.sqrt(2 / float(num_dim))
        self.W = np.random.normal(0.0, limit, size=(num_emb,num_dim))
        
    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array[int]): [# of batch, # of vocab(int) ] # [ [1, 3, 5, 10, 100 ...]  ]

        Returns:
            np.array: [# of batch, # of vocab, embedding_dim ] # [ [768] [768]                       ]
        """
        
        ################## edit here ###################
        output = self.W[x[:]] 
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
        
        print(expanded_d_prev)
        
        self.dW = np.sum(output, axis=0) 

        return output
        ################################################
    
    def __call__(self,x):
        return self.forward(x)

if __name__=="__main__":
    model = Embedding_np(10,20)
    x = np.random.randint(0,9, size=(1,5))
    output = model(x)
    model.backward(output)