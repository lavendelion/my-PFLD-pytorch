# my-PFLD-pytorch

Implementation of PFLD(Paper: "A Practical Facial Landmark Detector") by pytorch.

## 1. requirements

pytorch v1.1.0

torchvision v0.4.0

numpy v1.16.2

opencv



## 2. Data Preparation

* Download WFLW Dataset from [here](https://wywu.github.io/projects/LAB/WFLW.html).

  Wider Facial Landmarks in-the-wild (WFLW) contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks. 

* Download WFLW annotation from [here](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz).
* Unzip above two packages and put them on `./dataPrepare/WFLW/`
* move `Mirror98.txt` to ``./dataPrepare/WFLW/WFLW_annotations`
* run `./dataPrepare/SetPreparation.py` and `./dataPrepare/transform_data.py` sequentially.



## 3. train

* run `./train.py --dataset_dir ./dataPrepare`. You might change param '--dataset_dir' if you unzip datasets to another dir.
* Training log information will be saved in `./checkpoints/log.txt` by default.
* Weight of model will also be saved in `./checkpoints` every 5 epoch by default.



## 4. test

* run `./test.py --dataset_dir ./dataPrepare/test_data/imgs`.
* You can put your imgs in `./dataPrepare/test_data/imgs` to test your imgs.



## 5. reference

[PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)

[PFLD-Pytorch](https://github.com/polarisZhao/PFLD-pytorch)





