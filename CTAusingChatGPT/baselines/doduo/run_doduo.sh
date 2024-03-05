FOLDER=$1
NUM_CLASS=$2
NUM_CLASS_CPA=$3
SEED=$4
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES="2" python doduo/train_multi.py \
    --folder=$FOLDER \
    --max_length=32 \
    --batch_size=32 \
    --random_seed=$SEED \
    --num_classes=$NUM_CLASS \
    --num_classes_cpa=$NUM_CLASS_CPA \
    --colpair \