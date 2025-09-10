# Unsupervised Continual ConvNeXt for Visual Industrial Anomaly Detection
<img width="926" height="360" alt="image" src="https://github.com/user-attachments/assets/59dbcaeb-6e28-4ab7-a39b-7ff2c67179e5" />


## Introduction

Backbone: convnext_base_in22ft1k (ImageNet-22k finetuned), kept frozen

Dual-Prompt:
- Kernel prompt (additive delta on DWConv weights)
- Mask prompt (FiLM-style spatial modulation with bounded tanh + learnable scale)

KPK memory:
- Keys (from frozen model) identify task at test time
- Prompts (per-task) adapt features
- Knowledge (coreset of normal features) enables PatchCore kNN scoring

SCL with SAM: Structure-based Contrastive Learning uses SAM masks to pull same-structure patches together and push different structures apart.

Setting: 15 sequential MVTec AD tasks; no task IDs provided at inference.

## environment

### basic
python>=3.8, torch>=1.12, CUDA>=11.3, timm==0.6.7

### install SAM:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
or clone the repository locally and install with
```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

## prepare for training
rename the dataset dir to 'mvtec2d' and create SAM semantic directory
(processed mvtec2d-sam-b.zip is provided in repository)
```
cp -r $mvtec_origin_data_path('./mvtec2d') $mvtec_data_path('./mvtec2d-sam-b')
cd segment_anything
python3 dataset_sam.py --sam_type 'vit_b' --sam_checkpoint $your_sam_path --data_path $mvtec_data_path
```

## Training and Evaluation
Environment prepare:
```
datapath=/hhd3/m3lab/data/mvtec2d datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
```

<!-- datapath=/hhd3/m3lab/data/visa datasets=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum') -->
Training:
```
CUDA_VISIBLE_DEVICES=0 python3 run_ucad.py \
  --gpu 0 \
  --seed 0 \
  --memory_size 196 \
  --epochs_num 25 \
  --log_group IM224_UCAD_L5_P01_D1024_M196 \
  --save_segmentation_images \
  --log_project MVTecAD_Results \
  results ucad \
  -b wideresnet50 \
  -le layer2 -le layer3 \
  --faiss_on_gpu \
  --pretrain_embed_dimension 1024 \
  --target_embed_dimension 1024 \
  --anomaly_scorer_num_nn 1 \
  --patchsize 1 \
  --prompt_inject s4b1,s4b2,s4b3 \
  --analysis_site s4b3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset \
  --resize 224 \
  --imagesize 224 \
  "${dataset_flags[@]}" \
  mvtec $datapath
```

### Parameter
Main contents are contained in three files: ./patchcore/patchcore.py, ./patchcore/vision_transformer.py, and ./run_ucad.py.
Whether to save the image, the image size, and the memory size can all be modified in the above training command.

Training will directly provide the final results, and the inference process merely repeats this step. The final output will consist of two parts, with the lower metrics representing the final results, and the difference between them and the higher metrics results is denoted as FM.

## Acknowledgments

Our benchmark is built on [UCAD](https://github.com/shirowalker/UCAD), thanks for their extraordinary works!
