export DATA_DIR=/afs/inf.ed.ac.uk/group/project/material/querysum/data/squad/proc
export OUTPUT_DIR=/afs/inf.ed.ac.uk/group/project/material/querysum/model
export TASK_NAME=qas

#/afs/inf.ed.ac.uk/group/project/material/querysum/bin/python3 ./examples/run_glue.py \
python ./examples/my_run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR
