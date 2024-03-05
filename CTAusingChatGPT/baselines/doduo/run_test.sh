ARGS=$1
FOLDER=$2
NUM_CLASS=$3
NUM_CLASS_CPA=$4
CUDA_VISIBLE_DEVICES="2" python doduo/predict_multi.py \
    $1 \
    $2 \
    $3 \
    $4