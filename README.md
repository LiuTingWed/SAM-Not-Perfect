# Segment Anything Is Not Always Perfect
Code repository for our paper titled "[Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-World Applications](https://arxiv.org/abs/2304.05750)" (CVPRW Oral). 

![avatar](https://github.com/LiuTingWed/SAM-Not-Perfect/blob/main/sample.png)

------

## Updates
+ [x] This paper has been selected as **Most Inspective Paper**.
+ [x] Evaluation code has been released.
+ [x] This paper has been accepted as an *Oral Presentation* at the *CVPR'23 VISION Workshop*.

-------

## Get Started
### Eval SAM in different dataset
1. Download the **vit_b, vit_h and vim_l** model form https://github.com/facebookresearch/segment-anything then put these models to the **model_ck** folder.
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

## Citation
If you find this work useful for your research or applications, please cite using this BibTeX:
```bibtex
@article{ji2023segment,
  title={Segment anything is not always perfect: An investigation of sam on different real-world applications},
  author={Ji, Wei and Li, Jingjing and Bi, Qi and Liu, Tingwei and Li, Wenbo and Cheng, Li},
  journal={Computer Vision and Pattern Recognition Workshop (CVPRW)},
  year={2023}
}
```

## Acknowledgement

Thanks for the efforts of the authors involved in the [Segment Anything](https://github.com/facebookresearch/segment-anything). 
