<p align="center">
<h1 align="center">PinPoint3D: Fine-Grained 3D Part Segmentation from a Few Clicks</h1>
<p align="center">
<a href="https://github.com/Quit123?tab=repositories"><strong>Bojun Zhang</strong></a>
,
<a href="https://github.com/jian-77"><strong>Hangjian Ye</strong></a>
,
<a href="https://github.com/zh-plus"><strong>Hao Zheng</strong></a>
,
<a href="https://github.com/Cara-Zinc"><strong>Jianzheng Huang</strong></a>
<br>
<a href="https://pinpoint3d.online"><strong>Zhengyu Lin</strong></a>
, 
<a href="https://pinpoint3d.online"><strong>Zhenhong Guo</strong></a>
,
<a href="https://pinpoint3d.online"><strong>Feng Zheng</strong></a>
</p>
<!-- <h2 align="center">ICLR 2024</h2> -->
<h3 align="center"><a href="https://arxiv.org/abs/2509.25970">Paper</a> | <a href="https://pinpoint3d.online">Project Webpage</a></h3>
</p>
<p align="center">
<img src="./imgs/teaser.gif" width="500"/>
</p>
<p align="center">
<strong>PinPoint3D</strong> supports interactive multi-granularity 3D segmentation, where a user provides point clicks to obtain both object- and part-level masks efficiently in sparse scene point clouds.
</p>

## Installation 🔨

For training and evaluation, please follow the [installation.md](https://github.com/Quit123/PinPoint3D/blob/main/installation.md) to set up the environments.

## Interactive Tool 🎮

Please visit [interSeg3D-Studio
](https://github.com/SpatialtemporalAI/interSeg3D-Studio/tree/pinpoint3d) to experience the interactive annotation tool. It is a professional annotation platform designed specifically for the PinPoint3D model.

<p align="center">
<img src="./assets/demo.gif" width="75%" />
</p>

## Training 🚀

We design a new integrated dataset, PartScan, by integrating PartNet and ScanNet, and leveraging [PartField](https://github.com/nv-tlabs/PartField) to obtain part-level masks from object point clouds in ScanNet, thereby enhancing the generalization capability of PinPoint3D. You can download PartScan from [here(link gap)](https://drive.google.com/file/d/1Rg2JDjh8iFGKwzP0UMLBCkce7bCvO5-D/view?usp=sharing).

The command for training PinPoint3D with iterative training on PartScan is as follows:

```shell
./scripts/train_multi.sh
```


## Evaluation 📊

There are two datasets we provide for evaluation. Firstly, PartScan, a specialized dataset that integrates PartNet with ScanNet, and we evaluate the IoU of the original PartNet part masks within real-world ScanNet scenes. The second is MultiScan, which shows relatively modest results due to its coarser part granularity. You can download MultiScan from You can download MultiScan from [here(link gap)](https://drive.google.com/file/d/1Rg2JDjh8iFGKwzP0UMLBCkce7bCvO5-D/view?usp=sharing).

We provide the csv result files in the results folder, which can be directly fed into the evaluator for metric calculation. If you want to run the inference and do the evaluation yourself, download the pretrained [model](https://drive.google.com/file/d/1Rg2JDjh8iFGKwzP0UMLBCkce7bCvO5-D/view?usp=sharing) and move it to the weights folder. Then run:

### Evaluation on interactive multi parts 3D segmentation in multi-object:

- PartNet in Scene:
```shell
./scripts/eval_part.sh
```

### Evaluation on interactive multi parts 3D segmentation in single-object:

- PartNet in Scene:
```shell
./scripts/eval_part.sh
```

## Citation 🎓

If you find our code or paper useful, please cite:

```shell

@misc{zhang2025pinpoint3dfinegrained3dsegmentation,
title={PinPoint3D: Fine-Grained 3D Part Segmentation from a Few Clicks}, 
author={Bojun Zhang and Hangjian Ye and Hao Zheng and Jianzheng Huang and Zhengyu Lin and Zhenhong Guo and Feng Zheng},
year={2025},
eprint={2509.25970},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2509.25970},
}

```

