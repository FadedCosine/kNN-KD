DSTORE_SIZE=3949106
TEXT=iwslt14.tokenized.de_en
MODEL_PATH=checkpoints/$TEXT/base_mt/checkpoint_best.pt
DATA_PATH=../data/$TEXT/
DATASTORE_PATH=dstore/$TEXT

OUTPUT_PATH=generations

for lambda in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for knn_k in 4, 8, 16, 32, 64, 128, 256, 512, 1024
    do
        for temperature in 1, 10, 50, 100, 200, 500, 1000
        do
            echo "Do evaluation on: lambda ${lambda}, knn_k {$knn_k}, temperature {$temperature}..."
            CUDA_VISIBLE_DEVICES=0 python experimental_generate.py $DATA_PATH \
                --gen-subset test \
                --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
                --beam 5 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
                --batch-size 80 \
                --remove-bpe --quiet \
                --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
                'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $knn_k, 'probe': 32,
                'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
                'knn_lambda_type': 'fix', 'knn_lambda_value': $lambda, 'knn_temperature_type': 'fix', 'knn_temperature_value': $temperature,
                }"
        done
    done  
done


# DSTORE_SIZE=6903141
# TEXT=medical
# MODEL_PATH=checkpoints/wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt
# DATA_PATH=../data/law/
# DATASTORE_PATH=dstore/$TEXT

# OUTPUT_PATH=generations

# lambda=0.8
# knn_k=4
# temperature=10
# echo "Do evaluation on: lambda ${lambda}, knn_k {$knn_k}, temperature {$temperature}..."
# CUDA_VISIBLE_DEVICES=0 python experimental_generate.py $DATA_PATH \
#     --gen-subset test \
#     --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
#     --beam 5 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
#     --tokenizer moses --scoring sacrebleu \
#     --max-tokens 4096 \
#     --remove-bpe --quiet \
#     --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True,
#     'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True, 'k': $knn_k, 'probe': 32,
#     'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
#     'knn_lambda_type': 'fix', 'knn_lambda_value': $lambda, 'knn_temperature_type': 'fix', 'knn_temperature_value': $temperature,
#     }"