#!/bin/bash

# based on Nematus, https://github.com/rsennrich/nematus


logfile=run.log
printf "\n\n\n\nNew running from runwrap.sh\n" >> $logfile
echo "Time: $(date)." >> $logfile
printf "\n" >> $logfile
./run.sh 17000 50000 3000 26208 >>$logfile 2>&1
printf "\n" >> $logfile
echo "Time: $(date)." >> $logfile
printf "\n" >> $logfile




logfile=test.log
printf "\n\n\n\nNew running from runwrap.sh\n" >> $logfile
echo "Time: $(date)." >> $logfile
printf "\n" >> $logfile
./test.sh "models/model.26208.17000.50000.iter120000.npz models/model.26208.17000.50000.iter150000.npz models/model.26208.17000.50000.iter180000.npz models/model.26208.17000.50000.iter210000.npz" data/test.3000.diff data/test.3000.diff.ensemble.50k.50k.onlydobjs.nobrackets.output data/test.3000.msg models/model.26208.17000.50000_ensemble_test_bleu_score >>$logfile 2>&1
printf "\n" >> $logfile
echo "Time: $(date)." >> $logfile
printf "\n" >> $logfile
