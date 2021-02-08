# DeepSpeed GPT-2 Setup Guide

## 1. Datasets

### 1.1 Download from S3

I have prepared Wiki dataset and uploaded to S3.  URI s3://gpt-2-related/data/

all.json -- Dataset in loose Json format (not used in training)

[my-gpt2_text_document.bin](https://s3.console.aws.amazon.com/s3/object/gpt-2-related?region=us-east-1&prefix=data/my-gpt2_text_document.bin) [my-gpt2_text_document.idx](https://s3.console.aws.amazon.com/s3/object/gpt-2-related?region=us-east-1&prefix=data/my-gpt2_text_document.idx) -- Preprocessed data

[roberta-large-mnli-merges.txt](https://s3.console.aws.amazon.com/s3/object/gpt-2-related?region=us-east-1&prefix=data/roberta-large-mnli-merges.txt) [roberta-large-mnli-vocab.json](https://s3.console.aws.amazon.com/s3/object/gpt-2-related?region=us-east-1&prefix=data/roberta-large-mnli-vocab.json) -- Required files for GPT2 model. (Replace them and preprocess again if you don't trust the [source](https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/tokenization_roberta.html). But it works for now.)



### 1.2 Prepare from scratch

Check end of page of https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM-v1.1.5-3D_parallelism

Check https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/tokenization_roberta.html for necessary files i.e. merge file and vocab file

## 2. Run script

Go to https://github.com/redhairdragon/model_parallelism, inside **Megatron-LM-v1.1.5-3D_parallelism** folder, edit [test.sh](https://github.com/redhairdragon/model_parallelism/blob/main/Megatron-LM-v1.1.5-3D_parallelism/test.sh)

### 2.1 Set variables in the script

set GPUS_PER_NODE, NNODES

set DATA_PATH VOCAB_PATH MERGE_PATH

set NLAYERS, NHIDDEN, BATCHSIZE

### 2.2 Create hostfile

i.e 

ubuntu@172.31.8.37 slots=1
ubuntu@172.31.3.134 slots=1

### 2.3

Run test.sh

You can always go to aws->ec2-> launch template 