python OpenNMT-py/preprocess.py \
    -train_src NMT1/TrainingSet/train.26208.diff \
    -train_tgt NMT1/TrainingSet/train.26208.msg \
    -valid_src NMT1/TrainingSet/valid.3000.diff \
    -valid_tgt NMT1/TrainingSet/valid.3000.msg \
    -save_data model/demo \
    -src_seq_length 100 \
    -tgt_seq_length 30 \
    -lower \
#    -shuffle 1 # This was not done by the paper, but should yield the same results
