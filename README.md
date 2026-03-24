# Transformer
Transformers are neural networks that are very good at understanding and generating sequences, 
such as sentences. Unlike other models, such as RNNs, they do not read input one step at a time. 
Instead, they look at the entire sequence all at once. 
Context vectors were used in older transformers. The encoder reads the input, and produces an array called the context vector. It summarizes the importance of the input sequence. It became a bottleneck, since its a single fixed-size vector.

The context vector was replaced by attention, which solves the bottleneck issue. Instead of just sending only the last hidden state, the encoder sends all of the hidden states.
Transfomers use self attention to decide which parts of the input are the most important. 

[https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/]
