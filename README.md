# Nearest Neighbor Knowledge Distillation for Neural Machine Translation

Code for our NAACL 2022 paper **Nearest Neighbor Knowledge Distillation for Neural Machine Translation**. 

| Table of Contents |
|-|
| [Setup](#setup)|
| [Training](#training)|
| [Inference](#inference)|
| [Citation](#citation)|


This project implements our Adaptive kNN-MT as well as Vanilla kNN-MT.
The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [Adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt), many thanks to the authors for making their code avaliable.

## **Setup**
------
### **Dependencies**

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.7.1
* pytorch_scatter = 2.0.7
* 1.19.0 <= numpy < 1.20.0

### **Data preprocess**

For IWSLT'14 German to English data, download and preprocess the data follow [this instruction](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-iwslt14.sh).

For IWSLT'15 English to Vietnamese data, download the preprocessed data follow from [this site](https://nlp.stanford.edu/projects/nmt/). And as the suggestions in [this issue](https://github.com/pytorch/fairseq/issues/458), we also learn BPE for this data by:
```bash
sh commands/prepare_iwslt15.sh
``` 

For the multi-domain data, the raw data can be downloaded in [this site](https://github.com/roeeaharoni/unsupervised-domain-clusters). For convenience, we use the pre-processed dataset provided by [Adaptive kNN-MT' authors](https://github.com/zhengxxn/adaptive-knn-mt). We really appreciate their contributions.

## **Training**
------
### **Pre-trained Model and Data**
When building datastore, we need a pre-trained NMT model to map the translation contexts into respresentations.

For IWSLT datasets, we train a NMT model, which will be used to build the datastore, and in the meantime, act as a the base NMT model.

We preprocess the IWSLT data by fairseq:
```bash
sh commands/preprocess_data.sh
```

Then, train a Transformer model as the base NMT model:
```bash
sh commands/train_mt.sh
```

For the multi-domain data, we follow [Adaptive kNN-MT](<https://aclanthology.org/2021.acl-short.47.pdf>) to download the pre-trained translation model from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md).
We use the De->En Single Model for all domains datasets.


### **Build Datastore**
Run the following script to create datastore (includes key.npy and val.npy) for the data.
Please specify `DSTORE_SIZE`, `TEXT` , `MODEL_PATH`, `DATA_PATH`, and `DATASTORE_PATH`.
```bash
sh commands/build_datastore.sh
```

The DSTORE_SIZE depends on the num of tokens of target language train data. You can get it by two ways:

- find it in preprocess.log file, which is created by fairseq-process and in data binary folder.
- calculate wc -l + wc -w of raw data file.

The datastore sizes we used in our paper are listed as below:

| IWSLT'14 De-En | IWSLT'15 En-Vi  | Koran  | Medical      |Law      |
|---------|---------|--------|----------|----------|
| 3,949,106 | 3,581,500 | 524,374 | 6,903,141 |19,062,738 |

### **Build Faiss Index**

Run the following script to build faiss index for keys, which is used for fast knn search. when the knn_index is build, you can
remove keys.npy to save the hard disk space.

```bash
sh commands/build_faiss.sh
```
Note that, this process heavily rely on the [Faiss](https://github.com/facebookresearch/faiss). So if you got some error when runing the above scripts, be patient, and check the dependencies or version of your faiss. If you still stick in the mire, find some way out in [the issues of faiss](https://github.com/facebookresearch/faiss/issues)

### **kNN Search in Advance**
This part is one of the main points of our paper. As described in Section 3.1 of our paper, we conduct a kNN search for all the translation contexts in the training set immediately after the datastore building. Thus, for all the translation contexts in training set, we pre-store the retrival set that containing the k reasonble next token and corresponding distances.

For all the datasets, we save the retrieved results in bs1_knntargets_train_k64.mmp and bs1_fp16_distances_train_k64.mmp (or bs1_knntargets_train_k64.npy and bs1_fp16_distances_train_k64.npy if set --store-in-normal-npy, this means we store the retrieved results using normal `np.save`; else we use `np.memmap`) 
Run the following script:
```bash
sh commands/build_knntargets.sh
```

### **Train NMT Model with KNN-KD**
This part is another main point of our paper. We train the NMT model using our proposed KNN-KD. Run the following script:
```bash
sh commands/train_knnmt.sh
```
We recommend you to use below hyper-parameters to replicate the good results.

|             |  IWSLT'14 De-En | IWSLT'15 En-Vi  | Koran  | Medical      |Law      |
|:-----------:|:---:|:-------:|:---:|:-----:|:-----:|
|      k      |  64  |    64    |  16  |   4  | 4  |
| temperature |  100 |    100   |  100 |   10  |  10  |


The batch size and update-freq should be adjust by yourself depends on your gpu.

## **Inference**
------
### **Inference of Base NMT Model and our method**
Since our method maintain the same decoding process in standard Seq2Seq manner, the following scripts can be used to do the inference of base NMT model and our method:
```bash
sh commands/base_NMT_inference.sh
```
For multi-domains datasets, `--scoring sacrebleu --tokenizer moses` should be used for a fair comparsion with previous works.

### **Inference of Vanilla kNN-MT**

We also provide scripts to do vanilla kNN-MT inference
```bash
sh commands/vanilla_knn_mt_inference.sh
```
We recommend you to use below hyper-parameters to replicate the vanilla knn-mt results in our paper.

|             | IWSLT'14 De-En      | IWSLT'15 En-Vi  | Koran  | Medical | Law |
|:-----------:|:---:|:-------:|:---:|:-----:|:-----:|
|      k      |  8  |    16   |  16  |   4  | 4 |
|    lambda   | 0.3 |   0.4   | 0.8 |  0.8  | 0.8 |
| temperature |  100 |    100   |  100 |  10  | 10 |

## **Citation**
If you find our paper useful to your work, please kindly cite our paper:

```
@inproceedings{yang-etal-2022-nearest,
    title = "Nearest Neighbor Knowledge Distillation for Neural Machine Translation",
    author = "Yang, Zhixian  and
      Sun, Renliang  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.406",
    pages = "5546--5556",
    abstract = "k-nearest-neighbor machine translation ($k$NN-MT), proposed by Khandelwal et al. (2021), has achieved many state-of-the-art results in machine translation tasks. Although effective, $k$NN-MT requires conducting $k$NN searches through the large datastore for each decoding step during inference, prohibitively increasing the decoding cost and thus leading to the difficulty for the deployment in real-world applications. In this paper, we propose to move the time-consuming $k$NN search forward to the preprocessing phase, and then introduce $k$ Nearest Neighbor Knowledge Distillation ($k$NN-KD) that trains the base NMT model to directly learn the knowledge of $k$NN. Distilling knowledge retrieved by $k$NN can encourage the NMT model to take more reasonable target tokens into consideration, thus addressing the overcorrection problem. Extensive experimental results show that, the proposed method achieves consistent improvement over the state-of-the-art baselines including $k$NN-MT, while maintaining the same training and decoding speed as the standard NMT model.",
}
```
