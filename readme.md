# Part-Aware LSTM implemented in TensorFlow
**part-aware lstm** is proposed in [《NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis》](https://arxiv.org/abs/1604.02808), which is used for skeleton-based action recognition. It splits the whole body into 5 body part context. The specially designed lstm cell could better extract the context features than the normal one. The cell diagram is shown following:
![plstm][plstm]

Comparing with the normal lstm cell, it is not difficult to find that plstm has seperated i, g and g gate corresponding to different body part. 

It is inspired. So let us implement it in TensorFlow. 


[plstm]: ./imgs/plstm.png