#!/bin/bash
GPUS=$1
CONFIG=$2
MODEL=$3
PY_ARGS=${@:4}


python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    tools/convert.py -c ${CONFIG} --model ${MODEL} --model-config ${MODEL_CONFIG} ${PY_ARGS}
