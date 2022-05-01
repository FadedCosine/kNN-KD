TEXT=iwslt14.tokenized.de_en         
TrainKNNTarget_SIZE=3949106 
ValidKNNTarget_SIZE=178622
Strategy='knn_kd'
ETA=0
K=64
TEM=100

echo "Do training with ${Strategy} on: ETA ${ETA}, knn_k {$K}, temperature {$TEM}..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    ../data/$TEXT \
    --task translation_with_stored_knnls \
    --knn-k $K --save-k 1024 --train-knntarget-size $TrainKNNTarget_SIZE --valid-knntarget-size $ValidKNNTarget_SIZE \
    --knntarget-filename knntarget/$TEXT --knndistance-fp16 \
    --arch transformer_iwslt_de_en_with_datastore --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion knn_label_smoothed_cross_entropy --label-smoothing 0.1 --knn-temp $TEM --distil-strategy $Strategy \
    --max-tokens 8192 \
    --update-freq 1 \
    --seed 910 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 10 \
    --max-epoch 500 \
    --save-dir ~/knn_mt/checkpoints/$TEXT/knn$K''_temp$TEM''_KD \
    --fp16


# TEXT=law         
# TrainKNNTarget_SIZE=19062738 
# ValidKNNTarget_SIZE=82351
# Strategy='knn_kd'

# K=4
# TEM=10
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#     ../data/$TEXT \
#     --task translation_with_stored_knnls \
#     --knn-k $K --save-k 64 --train-knntarget-size $TrainKNNTarget_SIZE --valid-knntarget-size $ValidKNNTarget_SIZE \
#     --knntarget-filename knntarget/$TEXT --knndistance-fp16 \
#     --arch transformer_wmt19_de_en_with_datastore --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
#     --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --dropout 0.2 --weight-decay 0.0 \
#     --attention-dropout 0.1 --activation-dropout 0.1 \
#     --criterion knn_label_smoothed_cross_entropy --label-smoothing 0.1 --knn-temp $TEM --distil-strategy $Strategy \
#     --max-tokens 8192 \
#     --update-freq 1 \
#     --seed 910 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --keep-last-epochs 10 \
#     --max-epoch 10 \
#     --save-dir ~/knn_mt/checkpoints/$TEXT/knn$K''_temp$TEM''_time \
#     --fp16

