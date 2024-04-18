
# M2Doc


This is the pytorch implementation of Paper: M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis(AAAI 2024). The paper is available at [this link](https://ojs.aaai.org/index.php/AAAI/article/view/28552/29073).

<img src="demo/m2doc.png" width="100%">

## Installation
- Python=3.8.0
- transformers
- MMDetection
<!-- - OpenCV for visualization -->

## Steps
1. Install the repository (we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda create -n m2doc python=3.8 -y
conda activate m2doc
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
pip install transformers
git clone https://github.com/johnning2333/M2Doc.git
cd M2Doc
pip install -v -e .
```

2. dataset format
<!-- ```
datasets
|_ totaltext
|  |_ train_images
|  |_ test_images
|  |_ totaltext_train.json
|  |_ weak_voc_new.txt
|  |_ weak_voc_pair_list.txt
|_ mlt2017
|  |_ train_images
|  |_ annotations/icdar_2017_mlt.json
.......
``` -->

3. Train

4. Inference

<!-- ## Example results:

<img src="demo/results.png" width="100%"> -->

## Acknowlegement
[MMDetection](https://github.com/aim-uofa/AdelaiDet), [DINO](https://github.com/IDEA-Research/DINO), [VSR](https://github.com/hikopensource/DAVAR-Lab-OCR)

## Citation

If our paper helps your research, please cite it in your publications:

```BibText
@inproceedings{zhang2024m2doc,
  title={M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis},
  author={Zhang, Ning and Cheng, Hiuyi and Chen, Jiayu and Jiang, Zongyuan and Huang, Jun and Xue, Yang and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7233--7241},
  year={2024}
}
```

# Copyright

For commercial purpose usage, please contact Dr. Lianwen Jin: eelwjin@scut.edu.cn

Copyright 2019, Deep Learning and Vision Computing Lab, South China China University of Technology. http://www.dlvc-lab.net