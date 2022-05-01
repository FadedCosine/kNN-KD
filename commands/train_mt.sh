TEXT=iwslt14.tokenized_de_en
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    ../data/$TEXT/data-bin \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 300000 \
    --save-dir checkpoints/$TEXT/base_mt \
    --fp16

# TEXT=law 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#     ../data/$TEXT \
#     --arch transformer_wmt19_de_en_with_datastore --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
#     --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --dropout 0.2 --weight-decay 0.0 \
#     --attention-dropout 0.1 --activation-dropout 0.1 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 8192 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --keep-last-epochs 5 \
#     --max-epoch 10 \
#     --save-dir checkpoints/$TEXT/base_mt_time \
#     --fp16
