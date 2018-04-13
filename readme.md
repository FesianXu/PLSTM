# Part-Aware LSTM implemented in TensorFlow
**part-aware lstm** is proposed in [《NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis》](https://arxiv.org/abs/1604.02808), which is used for skeleton-based action recognition. It splits the whole body into 5 body part context. The specially designed lstm cell could better extract the context features than the normal one. The cell diagram is shown following:
![plstm][plstm]

Comparing with the normal lstm cell, it is not difficult to find that plstm has seperated i, g and g gate corresponding to different body part. 

It is inspired. So let us implement it in TensorFlow. 

Folder `network` includes three files, `PartAwareLSTMCell.py`,`DataLoader.py` and `PLSTM.py`. 
1. `PartAwareLSTMCell.py` is the Part-Aware LSTM cell.
2. `DataLoader.py` is the data loader used to load and preprocess the skeleton datas. Note that all skeleton data are formatted like **Array [Number_clips, {'mat':data, 'view': v, 'class':c, 'actor':a}]  data with shape of [Frames, 25, 3]**
3. `PLSTM.py` is the main train and evaluation entry.

Folder `utils` include one file, `gendata.py` which uses to generate the formatted data in numpy arrays. Note that all raw skeleton is stored in txt file and i transform them to mat file in MATLAB and i also save the (x,y,z) information.

## update
**13/4/2018**, add a jupyter notebook script in `app` folder used for training and evaluation. The code has not clear yet but could be a reference.

[plstm]: ./imgs/plstm.png