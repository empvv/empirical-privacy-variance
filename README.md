# Official repository for "Empirical Privacy Variance"

This repository contains the official implementation for the paper [Empirical Privacy Variance](https://openreview.net/pdf?id=oEvbe7vtOm), accepted at ICML 2025.

## Setup

Set up a conda environment with Python 3.11.0, and then install the required packages:

```bash
pip install -r requirements.txt
```

## Experiments on the Enron dataset

#### Dataset

Our processed dataset (see Appendix B.1) is provided in `exp_enron/enron_data`.

#### DP training

Our DP training code is built based on the [`dp-transformers`](https://github.com/microsoft/dp-transformers) package. 

Example command:
```bash
bash run_enron_SFT_dp.sh -c 1 -D enron_filtered_2 -b 32 -g 256 -l 1e-3 -a 0 -e 8 -x 1.1 -d 62 -f gpt2 -p 0.5 -s 250
```

#### Evaluation

1. Verbatim memorization ratio

Example command:
```bash
bash run_eval_enron_prompt_match.sh model-gpt2_data-enron_original_SFT_DP_eps-8_deltaexp-1.1_bs-8192_lr-5e-4_clip-0.5_lora-0_step-250_seed-42
```

2. Adversarial compression ratio

We use the [official repository](https://github.com/locuslab/acr-memorization/) of the paper [Rethinking LLM Memorization through the Lens of Adversarial Compression
](https://openreview.net/forum?id=KFmRMvzAZy).

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{
    hu2025empirical,
    title={Empirical Privacy Variance},
    author={Hu, Yuzheng and Wu, Fan and Xian, Ruicheng and Liu, Yuhang and Zakynthinou, Lydia and Kamath, Pritish and Zhang, Chiyuan and Forsyth, David},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=oEvbe7vtOm}
}
```