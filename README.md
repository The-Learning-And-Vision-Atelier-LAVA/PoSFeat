# Decoupling Makes Weakly Supervised Local Feature Better (PoSFeat)
This is the official implementation of **PoSFeat** (CVPR2022), a weakly supervised local feature training framework.

**Decoupling Makes Weakly Supervised Local Feature Better** <br>
[Kunhong Li](https://scholar.google.co.uk/citations?user=_kzDdx8AAAAJ&hl=zh-CN&oi=ao), [Longguang Wang](https://longguangwang.github.io/), [Li Liu](http://lilyliliu.com/Default.aspx), [Qing Ran](https://scholar.google.co.uk/citations?user=6ydy5oEAAAAJ&hl=zh-CN&oi=ao), [Kai Xu](http://kevinkaixu.net/index.html), [Yulan Guo* ](http://yulanguo.me/)<br>
**[[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Decoupling_Makes_Weakly_Supervised_Local_Feature_Better_CVPR_2022_paper.html)] [[Arxiv](https://arxiv.org/abs/2201.02861)] [[Blog](https://zhuanlan.zhihu.com/p/477818450)] [[Bilibili](https://www.bilibili.com/video/BV1xg411R7wD?spm_id_from=333.337.search-card.all.click)] [[Youtube](https://www.youtube.com/watch?v=VnjdkAOIndc)]** 

## Overview
We decoupled the description net training and detection net training, and postpone the detection net training. This simple but effective framework allows us to detect robust keypoints based on the optimized descriptors.
<p align="center"><img src="./imgs/framework.png" width="100%"> </p>

## Training
**(1) Download training data**

Down the preprocessed subset of MegaDepth from [CAPS](https://github.com/qianqianwang68/caps). If you want to test the local feature on [IMC](https://www.cs.ubc.ca/research/image-matching-challenge/current/), please manually remove the banned scenes (`0008 0021 0024 0063 1589`).
<!-- (<font style="background: #8B0000;" color=white> 0008 0021 0024 0063 1589 </font>). -->

**(2) Train the description net**

To start the description net training, please mannuly modify the `data_path` of `data_config_train` in [config/train_desc.yaml](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/configs/train_desc.yaml). 

Because of unknown reason, the multi-gpu training is really slow, so we should set single GPU available
```
export CUDA_VISIBLE_DEVICES=0
```

Then run the following command
```
python train.py --config ./configs/train_desc.yaml
```

It takes about 24 hours to finish description net training on a single NVIDIA RTX3090 GPU.

**(3) Train the detection net**

Similarly, modify the `datapath` and set single GPU available
```
export CUDA_VISIBLE_DEVICES=0
```
And run the command
```
python train.py --config ./configs/train_kp.yaml
```

**(4) The difference between the results trained with this code repo and in the paper**

In the paper, we use `SGD` optimizer with `lr=1e-3` to train the model, and here is the `Adam` with `lr=1e-4`. Note that, Adam with lr=1e-3 may not achieve convergence.

**(5) Multi-GPU training**

In this code repo, we use the `DistributedDataParallel` API of pytorch to achieve multi-GPU training, which is slow because of unknown reason. If you really need multi-gpu training, please modify the codes to use `DataParallel` API.

**(6) Visualization during training**

We also provide a visualization tool to give an intuition about the model performance during training. The results (including the heatmap, keypoints and raw matches) will be saved in the checkpoint path.
The visualization results includes the scoremap of keypoints (meaningless for description net training), the keypoints (sift for description net training) and matches (we color the match line with epipolar constraint).

**(7) Some dependencies**

We depend on the [path](https://path.readthedocs.io/en/latest/index.html) package to manage the paths in this repo, please follow the [readme on github](https://github.com/jaraco/path) or [introduction on PyPI](https://pypi.org/project/path/) to install it. Users may be familiar with other dependencies, you can simply use `pip` and `conda` to install dependencies.

## Evaluation
**(1) Feature extraction**

Using the `extract.py` can extract PoSFeat features. This file works with the [managers/extractor.py](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/managers/extractor.py), and users should provide a config file containing the datapath, detector config. The output can be `.npz` or `.h5`.

With `use_sift: True` in the config file, the output would be the sift keypoint with PoSFeat descriptor. The SIFT keypoints are detected with the OpenCV default settings in the dataloader.


**(2) HPatches**

We follow the evalutaion protocal proposed by [D2-Net](https://github.com/mihaidusmanu/d2-net/tree/master/hpatches_sequences) (please follow the introduction in D2-Net to download and modify the dataset), and modify the input codes for convenience. The result will be saved in [evaluations/hpatches/cache](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/tree/main/evaluations/hpatches/cache) as a `.npy` file, and we provide the results of several methods in the cache folder. Note that, you should mannuly remove the high resolution scenes in the original dataset.

Run the command
```
export CUDA_VISIBLE_DEVICES=0
python extract.py --config ./configs/extract_hpatches.yaml
```

Then turn to the [evaluations/hpatches](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/evaluations/hpatches) folder, modify the path in the evaluation script (if you donnot modify the script, there is only a PoSFeat_CVPR cache result) and run the script
```
cd ./evaluations/hpatches
python evaluation.py
```

When finishing the evaluation, you will get pictures of curves and a `.txt` file containing the quantitative results in the [evaluations/hpatches](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/evaluations/hpatches) folder.

<p align="center"><img src="./imgs/hpatches_res.png" width="100%"> </p>

**(3) Aachen-Day-Night**

We follow the standard Local feature challenge [pipeline](https://github.com/tsattler/visuallocalizationbenchmark) of [The Visual Localization Benchmark](https://www.visuallocalization.net/), please follow the introductions to download the dataset, then manage the data in this way
```
data_path_root_aachen
├── 3D-models/
│  ├── aachen_v_1/
│  │  ├── aachen_cvpr2018_db.nvm
│  │  └── database_intrinsics.txt
│  └── aachen_v_1_1/
│     ├── aachen_v_1_1.nvm
│     ├── cameras.bin
│     ├── database_intrinsics_v1_1.txt
│     ├── images.bin
│     ├── points3D.bin
│     └── project.ini
│ 
├── images # the v1 data and v1.1 data are mixed in this folder
│  └── images_upright/
│     ├── db/
│     ├── queries/
│     └── sequences/
│
├── queries/
│  ├── day_time_queries_with_intrinsics.txt
│  ├── night_time_queries_with_intrinsics.txt
│  └── night_time_queries_with_intrinsics_v1_1.txt
│ 
└── others/
   ├── database.db
   ├── database_v1_1.db
   ├── image_pairs_to_match.txt
   └── image_pairs_to_match_v1_1.txt
```

If you do not want to manage the data, you should mannuly modify the datapath settings in [evauluations/aachen/reconstruct_pipeline.py](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/evaluations/aachen/reconstruct_pipeline.py) (Line 329-339) and [evauluations/aachen/reconstruct_pipeline_v1_1.py](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat/blob/main/evaluations/aachen/reconstruct_pipeline_v1_1.py) (Line 319-330).

Before evaluation, we should extract the features first,
```
export CUDA_VISIBLE_DEVICES=0
python extract.py --config ./configs/extract_aachen.yaml
```

For evaulation on aachen-v1, run the command
```
cd ./evaluations/aachen
python reconstruct_pipeline.py --dataset_path [YOUR_data_path_root_aachen] \
--feature_path ../../ckpts/aachen/PoSFeat_mytrain/desc \
--colmap_path [YOUR_PATH_TO_COLMAP] \
--method_name PoSFeat_mytrain \
--match_list_path image_pairs_to_match.txt
```

For evaulation on aachen-v1.1, run the command
```
cd ./evaluations/aachen
python reconstruct_pipeline_v1_1.py --dataset_path [YOUR_data_path_root_aachen] \
--feature_path ../../ckpts/aachen/PoSFeat_mytrain/desc \
--colmap_path [YOUR_PATH_TO_COLMAP] \
--method_name PoSFeat_mytrain \
--match_list_path image_pairs_to_match_v1_1.txt
```

After evaluation, there will be 2 more folders created, `intermedia` contains intermediate results (such as sparse model and database) and `results` contains the `.txt` files that can be upload to the benchmark.

Note that, because the pose estimation (image registration) is based on the results of reconstruction, the results may be different each time.

**(4) ETH local feature benchmark**

Download the dataset following the introduction in [ETH local feature benchmark](https://github.com/ahojnnes/local-feature-evaluation) ([download instruction](https://github.com/ahojnnes/local-feature-evaluation/blob/master/INSTRUCTIONS.md)). Manage the dataset in this way
```
data_path_root_ETH_LFB
├── Alamo/
│  ├── images/
│  │  └── ...
│  └── database.db
│ 
├── ArtsQuad_dataset/
│  ├── images/
│  │  └── ...
│  └── database.db
│
├── Fountain/
│  ├── images/
│  │  └── ...
│  └── database.db
│ 
└── ...
```

Extract features first, we extract features for different scenes individually (mannuly modify the subfolder)
```
export CUDA_VISIBLE_DEVICES=0
python extract.py --config ./configs/extract_ETH.yaml
```

Then run evaluation for the scene
```
cd ./evaluations/ETH_local_feature
python reconstruction_pipeline.py --config ../../configs/extract_ETH.yaml
```

## BibeTeX

If you use this code in your project, please cite the following paper
```
@InProceedings{li2022decoupling,
    title={Decoupling Makes Weakly Supervised Local Feature Better},
    author={Li, Kunhong and Wang, Longguang and Liu, Li and Ran, Qing and Xu, Kai and Guo, Yulan},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022},
}
```