K=64
MODEL_PATH=/path/to/pretrained_model_path
DATA_PATH=/path/to/fairseq_preprocessed_data_path
DATASTORE_PATH=/path/to/saved_datastore
KNNTarget_PATH=/path/to/saved_knntargets
TEM=10
mkdir -p $KNNTarget_PATH

DSTORE_SIZE=3949106
KNNTarget_SIZE=178622

CUDA_VISIBLE_DEVICES=0 python save_knntargets.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation_build_knntargets \
    --valid-subset valid \
    --save-k $K \
    --path $MODEL_PATH \
    --batch-size 1 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 512 \
    --knntarget-size $KNNTarget_SIZE \
    --knn-temperature $TEM \
    --knntarget-mmap  $KNNTarget_PATH \
    --knndistance-fp16 \
    --seed 910 \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': False,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $K, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': 0.7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10,
     }"

DSTORE_SIZE=3949106
KNNTarget_SIZE=3949106

CUDA_VISIBLE_DEVICES=0 python save_knntargets.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation_build_knntargets \
    --valid-subset train \
    --save-k $K \
    --path $MODEL_PATH \
    --batch-size 1 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 512 \
    --knntarget-size $KNNTarget_SIZE \
    --knn-temperature $TEM \
    --knntarget-mmap  $KNNTarget_PATH \
    --knndistance-fp16 \
    --seed 910 \
    --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': False,
    'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $K, 'probe': 32,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': 0.7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10,
     }"