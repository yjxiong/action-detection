## Temporal Action Detection with Structured Segment Networks

### [Project Website][ssn]


**Note**: *We have released [MMAction](https://github.com/open-mmlab/mmaction), a full-fledged action understanding toolbox based on PyTorch. It includes implementation for SSN as well as other STOA frameworks for various tasks (action classification, temporal action detection, and spatial-temporal action detection). The lessons we learned in this repo are incorporated into MMAction to make it bettter. We highly recommend you switch to it. This repo will keep on being supported without further notice.*

*****

This repo holds the codes and models for the SSN framework presented on ICCV 2017

**Temporal Action Detection with Structured Segment Networks**
Yue Zhao, Yuanjun Xiong, Limin Wang, Zhirong Wu, Xiaoou Tang, Dahua Lin,  *ICCV 2017*, Venice, Italy.

[[Arxiv Preprint]](http://arxiv.org/abs/1704.06228)

A predecessor of the SSN framework was presented in
> **A Pursuit of Temporal Accuracy in General Activity Detection**
> Yuanjun Xiong, Yue Zhao, Limin Wang, Dahua Lin, and Xiaoou Tang, arXiv:1703.02716.



# Contents
----

* [Usage Guide](#usage-guide)
   * [Prerequisites](#prerequisites)
   * [Code and Data Preparation](#code-and-data-preparation)
      * [Get the code](#get-the-code)
      * [Download Datasets](#download-datasets)
      * [Pretrained Models](#pretrained-models)
   * [Extract Frames and Optical Flow Images](#extract-frames-and-optical-flow-images)
   * [Prepare the Proposal Lists](#prepare-the-proposal-lists)
   * [Train TAG Proposal Models](#train-tag-models)
   * [Testing Trained Models](#testing-trained-models)
      * [Evaluating on benchmark datasets](#evaluating-on-benchmark-datasets)
      * [Using reference models for evaluation](#using-reference-models-for-evaluation)
   * [Training SSN](#training-ssn)
      * [Training with ImageNet pretrained models](#training-with-imagenet-pretrained-models)
      * [Training with Kinetics pretrained models](#training-with-kinetics-pretrained-models)
 * [Temporal Action Detection Performance](#temporal-action-detection-performance)
    * [THUMOS14](#thumos14)
    * [ActivityNet v1.2](#activitynet-v12)
 * [Other Info](#other-info)
    * [Citation](#citation)
    * [Related Projects](#related-projects)
    * [Contact](#contact)


----
# Usage Guide

## Prerequisites
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

The training and testing in SSN is reimplemented in PyTorch for the ease of use. 
We need the following software to run SSN.

- Python3
- [PyTorch][pytorch]
- [DenseFlow][df] (for frame extraction & optical flow)
                   
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt

```

Actually, we recommend to setup the [temporal-segment-networks (TSN)][tsn] project prior to running SSN. 
It will help dealing with a lot of dependency issues of DenseFlow
However this is optional, because we will be only using the DenseFlow tool.

GPUs are required to for optical flow extraction and running SSN. 
Usually 4 to 8 GPUs in a node would ensure a smooth training experience.
 
 
## Code and Data Preparation
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

### Get the code
From now on we assume you have already set up PyTorch and had the DenseFlow tool ready from the [TSN][tsn] project.

Clone this repo with git, **please remember to use --recursive**

```bash
git clone --recursive https://github.com/yjxiong/action-detection

```

### Download Datasets

We support experimenting with two publicly available datasets for 
temporal action detection: THUMOS14 & ActivityNet v1.2. Here are some steps to download these two datasets.

- THUMOS14: We need the validation videos for training and testing videos for testing. 
You can download them from the [THUMOS14 challenge website][thumos14].
- ActivityNet v1.2: this dataset is provided in the form of YouTube URL list. 
You can use the [official ActivityNet downloader][anet_down] to download videos from the YouTube. 

After downloading the videos for each dataset, unzip them in a folder `SRC_FOLDER`.

### Pretrained Models

We provide the pretrained reference models and initialization models in standard PyTorch format. There is no need to manually download the initialization models. They will be downloaded by the `torch.model_zoo` tool when necessary.


## Extract Frames and Optical Flow Images

To run the training and testing, we need to decompose the video into frames. Also the temporal stream networks need optical flow or warped optical flow images for input. 

We suggest using the tools provided in the [TSN repo][tsn] for this purpose. Following instructions are from the [TSN repo][tsn]

> These can be achieved with the script `scripts/extract_optical_flow.sh`. The script has three arguments
> - `SRC_FOLDER` points to the folder where you put the video dataset
> - `OUT_FOLDER` points to the root folder where the extracted frames and optical images will be put in
> - `NUM_WORKER` specifies the number of GPU to use in parallel for flow extraction, must be larger than 1
> 
> The command for running optical flow extraction is as follows
> 
> ```
> bash scripts/extract_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER
> ```

## Prepare the Proposal Lists

Training and testing of SSN models rely on the files call "proposal lists". 
It records the information of temporal action proposals for videos together with the that of the groundtruth action instances.

In the sense that decoders on different machines may output different number of frames. 
We provide the proposal lists in a normalized form. 
To start training and testing, one needs to adapt the proposal lists to the actual number of frames extracted for each video.
To do this, run 

```bash
python gen_proposal_list.py DATASET FRAMES_PATH

```


## Train TAG models
Due to a large amount of inquiry for the training of TAG, we provide the following procedures to train binary actionness classifiers and generate proposals. 

### Generate sliding window proposals 
First of all, we generate a series of sliding-window proposals.

- THUMOS14
```bash
python gen_sliding_window_proposals.py validation rgb FRAME_PATH data/thumos14_sw_val_proposal_list.txt --dataset thumos14 
python gen_sliding_window_proposals.py testing rgb FRAME_PATH data/thumos14_sw_test_proposal_list.txt --dataset thumos14 
```

- ActivityNet v1.2
```bash
python gen_sliding_window_proposals.py training rgb FRAME_PATH data/activitynet1.2_sw_train_proposal_list.txt --dataset activitynet --version 1.2
python gen_sliding_window_proposals.py validation rgb FRAME_PATH data/activitynet1.2_sw_val_proposal_list.txt --dataset activitynet --version 1.2
```


### Training binary actionness classifier
Using the above proposals, we can train a binary actionness classifier.

```bash
python binary_train.py thumos14 MODALITY -b 16 --lr_steps 20 40 --epochs 45 
```

or

```bash
python binary_train.py activitynet1.2 MODALITY -b 16 --lr_steps 3 6 --epochs 7 
```

### Obtaining actionness score

Pretrained actionness classifier on THUMOS14 can be downloaded from 
[RGB Actionness Model](https://s3-us-west-2.amazonaws.com/awscv-public-data/ssn-actionness/binaryclassifier_thumos14_BNInception_rgb_checkpoint.pth.tar) and 
[Flow Actionness Model](https://s3-us-west-2.amazonaws.com/awscv-public-data/ssn-actionness/binaryclassifier_thumos14_BNInception_flow_checkpoint.pth.tar)

```bash
python binary_test.py DATASET MODALITY SUBSET TRAINING_CHECKPOINT ACTIONNESS_RESULT_PICKLE 
```

### Generating TAG proposals
THUMOS14
```bash
python gen_bottom_up_proposals.py ACTIONNESS_RESULT_PICKLE --dataset thumos14 --subset validation  --write_proposals data/thumos14_tag_val_proposal_list.txt  --frame_path FRAME_PATH
python gen_bottom_up_proposals.py ACTIONNESS_RESULT_PICKLE --dataset thumos14 --subset testing  --write_proposals data/thumos14_tag_test_proposal_list.txt  --frame_path FRAME_PATH
``` 
ActivityNet1.2
```bash
python gen_bottom_up_proposals.py ACTIONNESS_RESULT_PICKLE --dataset activitynet --subset training  --write_proposals data/activitynet1.2_tag_train_proposal_list.txt  --frame_path FRAME_PATH
python gen_bottom_up_proposals.py ACTIONNESS_RESULT_PICKLE --dataset activitynet --subset validation  --write_proposals data/activitynet1.2_tag_val_proposal_list.txt  --frame_path FRAME_PATH
```

where `ACTIONNESS_RESULTS_PICKLE` can be multiple (e.g. actionness predicted from both streams)

## Testing Trained Models
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

### Evaluating on benchmark datasets

There are two steps to evaluate temporal action detection with our pretrained models.

First, we will extract the detection scores for all the proposals by running 

```bash
python ssn_test.py DATASET MODALITY TRAINING_CHECKPOINT RESULT_PICKLE

```

Then using the proposal scores we evaluate the detection performance by running

```bash
python eval_detection_results.py DATASET RESULT_PICKLE

```

This script will report the detection performance in terms of [mean average precision][map] at different IoU thresholds.

### Using reference models for evaluation

We provide the trained models on our machines so you can test them before actual training any model. You can see the performance of the reference models in the [performance section](#temporal-action-detection-performance).

To use these models, run the following command
```bash
python ssn_test.py DATASET MODALITY none RESULT_PICKLE --use_reference

```

Addtionally, we provide the models trained with Kinetics pretraining, to use them, run
```bash
python ssn_test.py DATASET MODALITY none RESULT_PICKLE --use_kinetics_reference

```



## Training SSN
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

In the paper we report the results using pretraining on ImageNet. So we first iterate through this case.

### Training with ImageNet pretrained models

Use the following commands to train SSN

- THUMOS14

```bash
python ssn_train.py thumos14 MODALITY -b 16 --lr_steps 20 40 --epochs 45
```

- ActivityNet v1.2

```bash
python ssn_train.py activitynet1.2 MODALITY -b 16 --lr_steps 3 6 --epochs 7
```

Here, `MODALITY` can be `RGB` and `Flow`. `DATASET` can be `thumos14` and `activitynet1.2`. 
You can find more details about this script by running 
```bash
python ssn_train.py -h
```

After training, there will be a checkpoint file whose name contains the information about dataset, architecture, and modality.
This checkpoint file contains the trained model weights and can be used for testing.

### Training with Kinetics pretrained models

Additionally, we provide the initialization models pretrained on the Kinetics dataset. 
This pretraining process is known to boost the detection performance. 
More details can be found on [the pretrained model website][action_kinetics].

To use these pretrained models, append an option `--kin` to the training command, like

```bash
python ssn_train.py thumos14 MODALITY -b 16 --lr_steps 20 40 --epochs 45 --kin
```

and 

```bash
python ssn_train.py activitynet1.2 MODALITY -b 16 --lr_steps 3 6 --epochs 7 --kin
```

The system will use PyTorch's `model_zoo` utilities to download the pretrained models for you.


# Temporal Action Detection Performance
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

We provide a set of reference temporal action detection models. Their performance on benchmark datasets are as follow.
These results can also be found on the [project website][ssn]. You can download

### THUMOS14

| mAP@0.5IoU (%)                    | RGB   | Flow  | RGB+Flow      |
|-----------------------------------|-------|-------|---------------|
| BNInception                       | 16.18 | 22.50 | 27.36         |
| BNInception (Kinetics Pretrained) | 21.31 | 27.93 | 32.50         |
| InceptionV3                       | 18.28 | 23.30 | 28.00 (29.8*) |
| InceptionV3 (Kinetics Pretrained) | 22.12 | 30.51 | 33.15 (34.3*) |

\* We filter the detection results with the classification model from [UntrimmedNets][untrimmednets] to keep only those from the top-2 predicted action classes.

### ActivityNet v1.2

| Average mAP                       | RGB   | Flow  | RGB+Flow |
|-----------------------------------|-------|-------|----------|
| BNInception                       | 24.85 | 21.69 | 26.75    |
| BNInception (Kinetics Pretrained) | 27.53 | 28.0  | 28.57    |
| InceptionV3                       | 25.75 | 22.44 | 27.82    |
| InceptionV3 (Kinetics Pretrained) |       |       |          |

# Other Info
[[back to top](#temporal-action-detection-with-structured-segment-networks)]

## Citation


Please cite the following paper if you feel SSN useful to your research

```
@inproceedings{SSN2017ICCV,
  author    = {Yue Zhao and
               Yuanjun Xiong and
               Limin Wang and
               Zhirong Wu and
               Xiaoou Tang and
               Dahua Lin},
  title     = {Temporal Action Detection with Structured Segment Networks},
  booktitle   = {ICCV},
  year      = {2017},
}
```

## Related Projects
- [UntrimmmedNets][untrimmednets]: Our latest framework for learning action recognition models from untrimmed videos. (CVPR'17).
- [Kinetics Pretrained Models][action_kinetics] : TSN action recognition models trained on the Kinetics dataset.
- [TSN][tsn] : state of the art action recognition framework for trimmed videos. (ECCV'16).
- [CES-STAR@ActivityNet][anet] : winning solution for ActivityNet challenge 2016, based on TSN.
- [EnhancedMV][emv]: real-time action recognition using motion vectors in video encodings.

## Contact
For any question, please file an issue or contact
```
Yue Zhao: thuzhaoyue@gmail.com
Yuanjun Xiong: bitxiong@gmail.com
```



[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads
[tdd]:https://github.com/wanglimin/TDD
[anet]:https://github.com/yjxiong/anet2016-cuhk
[faq]:https://github.com/yjxiong/temporal-segment-networks/wiki/Frequently-Asked-Questions
[bs_line]:https://github.com/yjxiong/temporal-segment-networks/blob/master/models/ucf101/tsn_bn_inception_flow_train_val.prototxt#L8
[bug]:https://github.com/yjxiong/caffe/commit/c0d200ba0ed004edcfd387163395be7ea309dbc3
[tsn_site]:http://yjxiong.me/others/tsn/
[custom guide]:https://github.com/yjxiong/temporal-segment-networks/wiki/Working-on-custom-datasets.
[thumos14]:http://crcv.ucf.edu/THUMOS14/download.html
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[anet_down]:https://github.com/activitynet/ActivityNet/tree/master/Crawler
[map]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
[action_kinetics]:http://yjxiong.me/others/kinetics_action/
[pytorch]:https://github.com/pytorch/pytorch
[ssn]:http://yjxiong.me/others/ssn/
[untrimmednets]:https://github.com/wanglimin/UntrimmedNet
[emv]:https://github.com/zbwglory/MV-release

