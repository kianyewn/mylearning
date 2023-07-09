### References

1. Tokenizer details from HF (refer to this if you want to train own tokenizer)
    - https://huggingface.co/docs/tokenizers/pipeline
    
    
2. difference between autotokenizer.from_pretrained and specifictokenizer.from_pretrained
    - gs: huggingface autotokenizer different from tokenizer
    - https://github.com/huggingface/transformers/issues/5587
    

3. Why do we use offsets?
    - gs: huggingface tokenizer offsets
    -
https://huggingface.co/transformers/v4.2.2/custom_datasets.html#:~:text=For%20each%20sub%2Dtoken%20returned,its%20corresponding%20label%20to%20%2D100%20.


4.Difference between token ids and attention mask
- [ref](https://jaketae.github.io/category/common-sense/#:~:text=Input%20IDs%20are%20obvious%3A%20these,where%20two%20sentences%20are%20given.)

- input IDs are obvious: these are simply mappings between tokens and their respective IDs. The attention mask is to prevent the model from looking at padding tokens. The token type IDs are used typically in a next sentence prediction tasks, where two sentences are given.


5. Error from using TFRobertaModel
- No module named 'keras.engine'
- Due to most latest version of tensorflow ... 
- downloaded the tf_model.h5 and saved in current directory, but does work

6. What doe conv1d(filters=1,kernel_size=1) mean
- This link is not necessarily useful: https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610
- This means that we are using 1 filters (output channel), kernel_size=1 means specifies the length of the 1D convolution window.
- Convolution is applied on the second dimension. eg (4, 10, 128) -> convolution is applied on dim=1 (i.e shape=(10,)). 
    - To run sample, need to use the convolution weights
- There are differences in conv1d implementation in pytorch and tensorflow. Same thing for dropout1d
    - https://pytorch.org/docs/stable/generated/torch.nn.Dropout1d.html
    - To use appropriate dropout on the timestep, we need to convert data from (N,T,C) -> dropout((N, C, T)) -> (N,T,C)


7. When to use crossentropy loss and when to use binary cross entropy loss
- gs: (pytorch cross entropy loss vs binarycrossentropy loss)
- https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
    - BinaryCrossEntropy() loss should be used for binary, crossentropy loss should be used for multi-class classification
    - Reason:
       - crossentropy loss will take as input a target y that can take on values (0,C), and expects the predictions to be of shape (batch_size, num_classes)
        - BinaryCrossEntropy loss will take in as input a target y that can take on values (0,1), and expects the predictions to be of shape (batch_size, 1)
        - CrossEntropy can be used for binary classification. 
           - Using sigmoid
               - Final output is shape (batch_size, 2) because crossentropy loss needs to take in predictions with (bsize, num_classes)
               - This means that we use one sigmoid for each 0 and 1 prediction before feeding it into CrossEntropy loss. 
               - But output probabilities will be meaningless. eg σ([-2.34, 3.45])=[8.79%, 96.9%] does not make sense
           - Using softmax
               Final output is also of shape (batch_size, 2) because crossentropy loss requires predictions with shape (bsize,num_classes)
           - However, for binary classification where there are only two classes, the output from softmax tends to always be close to 0 and close to one. Eg. softmax([-2,34, 3,45])=[0.3%, 99.7%]
           - So softmax is only suitable for multi-class classification.
           
- When to use binarycrossentropy, crossentroppy loss
    - TLDR: BCE: multilabel, binary classification, CE: multiclass
- https://stackoverflow.com/questions/59336899/which-loss-function-and-metrics-to-use-for-multi-label-classification-with-very
    - gs: multi label classification binarycross entorpy
- https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
    - gs: crossentropy loss for multi label classification
- https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203
    - gs: loss function for multi label classification pytorch