# E.T.: Re-Thinking Self-Attention for Transformer Models on GPUs
E.T. accelerates the inference of Transformer-family models by introducing a novel self-attention architecture and an attention-aware pruning design, which takes advantage of GPU hardware and Transformer-specific optimization. 
## Platform 
- GPU: NVIDIA V100S, CUDA 11.1
- OS: Ubuntu 18.04.1
- Compiler: GCC 7.5.0, NVCC V11.1.105

## Install
### Dependency
- Python-3.6.9
- PyTorch-1.4.0
- Tensorflow-2.3
- transformers-3.5
- cnpy
- Anaconda3
### Pre-trained model
Hugging Face's Transformers:

https://github.com/huggingface/transformers
### Dataset
General Language Understanding Evaluation (GLUE) benchmark is used for training and evaluation.
https://gluebenchmark.com/tasks

### Init pruning submodule 
The pruning code is integrated by git submodule for simplicity.
(See: https://github.com/cctry/SCpaper-2021-pruning.git)
```
git submodule init
git submodule update
```
### Build CUDA code
``` 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
## Examples
There are two examples of using E.T.
### Transformer pruning & inference
The source code of this example is in the `./example` directory.

Step 1: Configure training environment
```bash
cd SCpaper-2021-pruning/transformer_code
conda env create -f pytorch_BCM_pruning.yml
```
Step 2: Pruning & Training
```bash
bash run_admm_whole_block_pruning_transformer.sh
```
Step 3: Inference demo
```bash
mv <model file> ../../
cd ../../
mkdir npyfile
python3 ./example/npy2pt.py --model_file <model file> --weight_folder ./npyfile/
./build/example/transformer_infer ./npyfile/weights.txt ./npyfile/ 
```
### BERT pruning on MRPC

Following is an example of how we run the experiment of pruning on MRPC

Step 1: Fine-tuning

```bash
cd SCpaper-2021-pruning/BERT_tensor/examples/text-classification
bash bert-test_glue_finetune_MRPC.sh
```

Step 2: Re-weighted training

```bash
bash bert-test_prune_MRPC_tensor_tile.sh
```

Step 3: Prune and Re-weighted retraining

```bash
bash bert-test_retrain_MRPC_tensor_tile.sh
```
Please refer to https://github.com/cctry/SCpaper-2021-pruning.git for more BERT pruning detail.
