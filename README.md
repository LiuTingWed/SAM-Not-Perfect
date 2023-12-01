# Segment Anything Is Not Always Perfect
Code repository for our paper titled "[Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-World Applications](https://arxiv.org/abs/2304.05750)" (CVPRW Oral). 

![avatar](https://github.com/LiuTingWed/SAM-Not-Perfect/blob/main/sample.png)

------

## Updates
+ [x] Long version of this work has been accepted by *Machine Intelligence Research*.
+ [x] This work is awarded as **[Best Paper](https://vision-based-industrial-inspection.github.io/cvpr-2023/)** (Most Insightful Paper) at the *CVPR'23 VISION Workshop*.
+ [x] Evaluation code has been released.
+ [x] This work has been accepted as an *Oral Presentation* at the *CVPR'23 VISION Workshop*.

-------

## Get Started
### Eval SAM in different dataset
1. Download the **vit_b, vit_h and vim_l** model from https://github.com/facebookresearch/segment-anything then put these models to the **model_ck** folder.
2. Prepared own datasets put into the **datasets** folder.
3. Set right path in /scripts/amg.py, then:
> run amg.py
### Chosen best results form the sam_output folder
1. After inferring, the SAM model generates predicted maps from a singer RGB image (**multimask_output=True**). Check right path in **sam_dice_f1_mae.py** or **sam_f1_dice_mae.py** to decide the best map selected by Dice or F1 metrics. 
### Eval other methods in different dataset
1. Prepared these methods predicted maps to put into the **other_methods_output** folder.
2. Check right path in /scripts/other_methods_dice_mae.py, then:
> run other_methods_dice_mae.py
-------


## Datasets

The download links of the dataset involved in our work are provided below.

DUTS | COME15K | VT1000 | DIS | COD10K | SBU | CDS2K | ColonDB 
 :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:
[Link](http://saliencydetection.net/duts/) | [Link](https://github.com/jingzhang617/cascaded_rgbd_sod) | [Link](https://github.com/lz118/RGBT-Salient-Object-Detection) | [Link](https://xuebinqin.github.io/dis/index.html) | [Link](https://dengpingfan.github.io/pages/COD.html) | [Link](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html) | [Link](https://github.com/DengPingFan/CSU) | [Link](http://vi.cvc.uab.es/colon-qa/cvccolondb/) 

-------

## Citation
If you find this work useful for your research or applications, please cite using this BibTeX:
```bibtex
@article{ji2023segment,
  title={Segment anything is not always perfect: An investigation of sam on different real-world applications},
  author={Ji, Wei and Li, Jingjing and Bi, Qi and Liu, Tingwei and Li, Wenbo and Cheng, Li},
  journal={arXiv preprint arXiv:2304.05750},
  year={2023}
}
```

## Acknowledgement

Thanks for the efforts of the authors involved in the [Segment Anything](https://github.com/facebookresearch/segment-anything). 
