# Classifier

A Pytorch implementation of a Deep Averaging Network (DAN) classifier. Uses pre-trained Glove embeddings. 

Hidden layer dimensions must be specified with the `--layers` flag, i.e `--layers 50 30`.

Other optional flags:

    --data_dir: the directory containing the data files
    --temp_dir: the directory for temporary files (i.e. saved model state, embeddings)
    --file_type: the tabular format (csv / tsv)
    --emb_dim: the size of the pretrained Glove embedding (50, 100, 200, 300)
    --batch_dims: the batch size of training and test sets
    --lr: the learning rate
    --num_steps: the number of training steps
    --num_processes: the number of parallel training processes (default = no parallelism)
    --baseline: compare with naive bayes baseline
    --plot: plot loss

Note that if `--num_processes` is greater than `1`, logging and `--plot` are not supported.

## TO DO

- [ ] Add logging for `torch.multiprocessing`
- [ ] Support ELMo contextual embeddings





