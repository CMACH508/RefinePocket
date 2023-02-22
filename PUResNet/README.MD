# PUResNet(Predicting protein-ligand binding sites using deep convolutional neural network)
This is an re-implementation of PUResNet [1] in PyTorch. Official implementation can be seen here [2]. We re-trained and tested our PUResNet on our train and test datasets.

# Train dataset
You can download our train dataset from here (https://jbox.sjtu.edu.cn/l/a1t6DP)

# Pre-trained model
You can download our pre-trained model from here (https://jbox.sjtu.edu.cn/l/a1t6DP)

# Train
If you want to train on your own train dataset, you can run the following command:
```
python main.py --data_root DATA_ROOT --gpu '0' --batch_size 5 --base_lr 0.001
```


# Test
If you want to test our pretrained model, you can run the following command:
```
### DCC DVO
python -u test_topn_dcc.py --model_path MODEL_PATH --batch_size 30 --test_set coach420 --is_dca 0 --top_n 100 --gpu 0

### DCA
python -u test_topn_dcc.py --model_path MODEL_PATH --batch_size 30 --test_set coach420 --is_dca 1 --top_n 2 --gpu 0

### TOP-N DCC DVO
python -u test_topn_dcc.py --model_path MODEL_PATH --batch_size 30 --test_set coach420 --is_dca 0 --top_n 0 --gpu 0
```



# Reference
[1] J. Kandel, H. Tayara, and K. T. Chong, “Puresnet: prediction of protein-ligand binding sites using deep residual neural network,” Journal of cheminformatics, vol. 13, no. 1, pp. 1–14, 2021

[2] https://github.com/jivankandel/PUResNet