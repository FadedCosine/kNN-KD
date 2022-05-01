DSTORE_SIZE=specific_size
TEXT=specific_dataset
MODEL_PATH=/path/to/pre-trained_model
DATA_PATH=/path/to/data
DATASTORE_PATH=/path/to/datastore

mkdir -p $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 25600 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 512 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH