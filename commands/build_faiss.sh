DSTORE_SIZE=specific_size
DSTORE_PATH=/path/to/datastore

CUDA_VISIBLE_DEVICES=0 python train_datastore_gpu.py \
  --dstore_mmap $DSTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --dstore-fp16 \
  --faiss_index ${DSTORE_PATH}/knn_index \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024
