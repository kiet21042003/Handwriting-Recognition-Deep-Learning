# Handwriting Recognition

## References:
- Spatial Transformer Networks: [link](https://arxiv.org/abs/1506.02025)
- ResNet extractor: [link](https://arxiv.org/abs/1512.03385)
- LSTM and Bidirectional LSTM: [link](https://arxiv.org/abs/1402.1128)
- Attention mechanism:
- Reference Repo: [clovaai](https://github.com/clovaai/deep-text-recognition-benchmark/tree/master)

## Dataset
IAM Handwritting database with some fundamental preprocessing for our pipeline. The original can be found [here](https://www.bing.com/ck/a?!&&p=d578fce1f73a878fJmltdHM9MTY5Njk4MjQwMCZpZ3VpZD0yYzc0ODExMS01OGM0LTZlZWItM2UyNS05MmI5NTkzMzZmNTUmaW5zaWQ9NTE4Mw&ptn=3&hsh=3&fclid=2c748111-58c4-6eeb-3e25-92b959336f55&psq=iam+handwriting+database&u=a1aHR0cHM6Ly9ma2kudGljLmhlaWEtZnIuY2gvZGF0YWJhc2VzL2lhbS1oYW5kd3JpdGluZy1kYXRhYmFzZQ&ntb=1)

## How to train
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/pretrain_1.sh
```
In order to change the configuration, create new bash file in [scripts](scripts/). To get all options for configuration, refer to [options](options.py)
