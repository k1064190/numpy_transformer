# numpy_transformer
numpy implementation of pytorch deep learning models including Transformer


# Completed Checklist

Here is the current checklist of completed items:

## Activation Functions
- ReLU (Rectified Linear Unit)
- Sigmoid

## Loss Functions
- Cross-Entropy Loss
- Binary Cross-Entropy Loss

## Layers
- Linear Layer
- CNN (Convolutional Neural Network)

## Optimizers
- SGD
- SGD with momentum
- Adam

## Utilities
- Dropout
- Embedding
- Positional Encoding
- Flatten
- MaxPooling


# Curriculum
## **Week 1 - Basis**
- **Topic**
  - Activation Functions (ReLU, Sigmoid)
  - Loss Functions (Cross-Entropy, Binary Cross-Entropy)
  - Linear Layer
- **Week 1 Exercise**
  - Compare NumPy implementation with Torch functions

## **Week 2 - CNN**
- **Topic**
  - Convolution Layer (2D)
  - MaxPooling (2D)
  - Flatten Layer
- **Week 2 Exercise**
  - Compare NumPy/Torch model accuracy with Linear/CNN models

## **Week 3 - Optimizer/Embedding**
- **Topic**
  - Dropout
  - Embedding
  - Positional Encoding
  - Optimizers (SGD, momentum, Adam)
- **Week 3 Exercise**
  - Compare NumPy model accuracy with different Optimizers

## **Week 4 - RNN/Tokenizer(TODO)**
- **Topic**
  - RNN
  - Word Tokenzizer
- **Week 4 Exercise**
  - Train Tokenizer

## **Week 5 - LSTM/Seq2Seq(TODO)**
- **Topic**
  - LSTM
  - Seq2Seq
- **Week 5 Exercise**
  - Train Chatbot with numpy model

## **Week 6 - Attention(TODO)**
- **Topic**
  - Attention
  - Normalization (Layer Normalization, Batch Normalization)
- **Week 6 Exercise**
  - Train Chatbot with numpy model (Advanced)





```
.
├── dummy_codes.py
├── README.md
├── exercise_codes
│   ├── week_1_relu-sigmoid-bceloss-mlp.py
│   ├── week_2_cnn_main.py
│   ├── week_2_models.py
│   ├── week_3_models.py
│   └── week_3_optimizer_main.py
├── numpy_models
│   ├── activations
│   │   ├── relu.py
│   │   └── sigmoid.py
│   ├── commons
│   │   ├── cnn.py
│   │   └── linear.py
│   ├── losses
│   │   ├── binary_ce.py
│   │   └── ce.py
│   ├── optimizer
│   │   ├── Adam.py
│   │   ├── SGD_momentum.py
│   │   └── SGD.py
│   └── utils
│       ├── dropout.py
│       ├── embedding.py
│       ├── pooling.py
│       ├── positional_encoding.py
│       └── flatten.py
└── torch_models
    ├── activations
    │   ├── relu.py
    │   └── sigmoid.py
    ├── commons
    │   ├── cnn.py
    │   └── linear.py
    ├── losses
    │   ├── binary_ce.py
    │   └── ce.py
    ├── optimizer
    │   ├── Adam.py
    │   ├── SGD_momentum.py
    │   └── SGD.py
    └── utils
        ├── dropout.py
        ├── embedding.py
        ├── pooling.py
        ├── positional_encoding.py
        └── flatten.py
```