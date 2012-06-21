#!/bin/sh

cd /nfshome/coman/NO_INB_BACKUP/Dropbox/Work/MastersThesis/v7/bin
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/opt/local-ubuntu/bin:/opt/local-ubuntu/bin
export JOB_ID=$1
export CONFIGURATION=$2
export TASK_ID=$3
export TRANSFER_IN_PATH=$4
export TRANSFER_OUT_PATH=$5
matlab -nojvm -nodisplay -r crossvalidate_par_worker
