TEXT=iwslt14.tokenized.de_en
DATA_PATH=../data/$TEXT
OUTPUT_PATH=generations
MODEL_PATH=/path/to/model
echo "Do evaluation with on {$MODEL_PATH}..."
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $DATA_PATH\
    --gen-subset test \
    --path $MODEL_PATH \
    --beam 5 --max-len-a 1.2 --max-len-b 10 \
    --max-tokens 4096 \
    --remove-bpe --quiet 


TEXT=law
MODEL_PATH=/path/to/model
DATA_PATH=../data/$TEXT
DATASTORE_PATH=dstore/$TEXT
OUTPUT_PATH=generations

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $DATA_PATH\
    --gen-subset test \
    --path $MODEL_PATH \
    --beam 5 \
    --remove-bpe --max-tokens 4096 --quiet \
    --tokenizer moses --scoring sacrebleu 