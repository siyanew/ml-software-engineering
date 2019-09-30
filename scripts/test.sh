#!/usr/bin/env bash
python OpenNMT-py/translate.py \
    -model model/demo-model_step_5000.pt  \
    -src NMT1/TestSet/test.3000.diff \
    -output model/pred.txt  \
    -replace_unk  \
    -verbose \
    -max_length 100 \
    -report_blue \
    -report_rouge \
