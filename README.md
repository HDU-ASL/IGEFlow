# IGEFlow: Implicit Guidance Image Enhancement for Low-Light Optical Flow Estimation

This project provides the official implementation of 'IGEFlow: Implicit Guidance Image Enhancement
for Low-Light Optical Flow Estimation'.

## Abstract

As a fundamental visual task, optical flow estimation has widespread applications in the field of computer vision. However, it faces significant challenges in low-light scenarios, where issues such as low signal-to-noise ratios make accurate optical flow estimation particularly difficult. Furthermore, applying style transfer models to enhance images degrade optical flow estimation results due to a lack of temporal geometric consistency.
In this paper, we propose a low-light optical flow method that employs implicit guidance for image enhancement. We utilize a channel-attention-based image enhancement network to improve the quality of low-light images, followed by an iterative optical flow method for flow computation. During training, the encoded features extracted from the enhanced images are supervised by features from a pre-trained network as well as by the optical flow task. This method allows the enhancement network to be implicitly guided by normal-light images and the specific subsequent tasks, enabling it to learn normal-light knowledge that enhances feature information suitable for optical flow estimation.Experiments conducted on both synthetic and real datasets demonstrate that our proposed method significantly improves performance on public low-light datasets.

## Environment

The code has been tested with PyTorch 2.0 and Cuda 11.1.

```Shell
conda env create -f env.yaml
```

## Required Data

To evaluate/train model, you will need to download the required datasets.

* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [VBOF_dataset](https://github.com/mf-zhang/Optical-Flow-in-the-Dark/tree/main/VBOF_dataset)

You need to use `syndata.py` to synthesize a noisy dataset based on the Flyingchairs dataset.
Download the raw format VBOF dataset, convert it to RGB format, and organize the format through `vbofdata.py`.

Then, download the pretrained weights [train_guide.pt](https://drive.google.com/file/d/1QyZgdYIjssOOzcqemxSgX1YDB8bjF5HQ/view?usp=sharing) and save it to `checkpoints`

Thanks to [Conference Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Optical_Flow_in_the_Dark_CVPR_2020_paper.pdf). The download link for the VBOF dataset: [VBOF](https://github.com/mf-zhang/Optical-Flow-in-the-Dark)

```Shell
├── datasets
    ├── FlyingChairs_release
        ├── data
    ├── VBOF
        ├── sony
        ├── fuji
        ├── nikon
	├── canon
```

## Evaluation

You can evaluate a trained model using `script`

```Shell
./options/eva/eva_raft_guide.sh
```

## Training

We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard

```Shell
./options/train_guide.sh
```

If you have a RTX GPU, training ca be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)

```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation

You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension

```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```

and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
