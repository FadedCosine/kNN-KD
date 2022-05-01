# fairseq preprocess
TEXT=/path/to/iwslt14_data
fairseq-preprocess --source-lang en --target-lang de  \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --joined-dictionary \
    --destdir /destdir/path/to/iwslt14_data_bin \
    --workers 20
TEXT=/path/to/iwslt15_data
fairseq-preprocess --source-lang en --target-lang vi \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --joined-dictionary \
    --destdir /destdir/path/to/iwslt15_data_bin \
    --workers 20