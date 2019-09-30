python OpenNMT-py/train.py \
    -data model/demo \
    -save_model model/demo-model \
    -gpu_ranks 0 \
    -learning_rate 0.0001 \ # Compied from the paper, OpenNMT suggest lrate=1.0 for adadelta
    -optim adadelta \
    -batch_size 80  \ # Batch sizes were defined as 80, it might be better to use multiple of 2^x
    -valid_batch_size 80 \
    -word_vec_size 512 \ # Embeddings sizes for both Enc and Dec
    -encoder_type brnn \
    -decoder_type brnn \
    -rnn_type LSTM \
    -rnn_size 1000 \ # Hidden state sizes (Enc and Dec) - This parameter was not set by the paper, 1000 is their framework default
    -layers 1024 \ # amount of hidden layers (Enc and Dec)
    -global_attention mlp \ # Addative attention from Bahdanau
    -dropout 0.1 \
    -epochs 5000 \
    -valid_steps 10000 \
    -save_checkout_steps 30000 \
    -keep_checkout -1 \
    -early_stopping 10 \ # Not supplied by paper, this is the default from their framework




