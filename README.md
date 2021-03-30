# UV-ABLM
Ubiquitous Vision of Transformers for Person Re-identification: A self-attention and aligned batch layoff module based novel person re-id approach

Pytorch based library "[torchreid](https://github.com/KaiyangZhou/deep-person-reid)" is used for this work, which provides a unified interface for numerous public person re-id benchmarks. 

# Evaluation
UV re-id trained models can be downloaded from [GoogleDriver](https://drive.google.com/drive/folders/1uHLHLJwf5NfvzZL9AwemCmsY6334DWPy?usp=sharing).

Copy the trained weights in the directory of "models" before running the evaluation script: python eval.py 


# Citations
If you find this code useful to your research, please cite the following papers.

@article{torchreid, title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch}, author={Zhou, Kaiyang and Xiang, Tao}, journal={arXiv preprint arXiv:1910.10093}, year={2019} }

@article{transformer,   title = {Attention is all you need}, author = {Vaswani et. al}, journal = {Advances in neural information processing systems}, year = {2017} }

