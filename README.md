# MEGNet
Using EEGNet and converting this into MEGNet for use with Magnetoenecalography Data. Includes a Preprocessing Pipeline, and Variations of EEGNet converted for use in MEG Datasets.

##########################################################

To note - Code is still a work in progress.

Pipeline may be subject to change.

Pipeline includes a variety of processing steps involved in the pre-processing stage of MEG data. As is often the case, MEG captures a high temporal spatial accuracy within recordings, however the noise to signal ratio (along with EEG) is often low. This pipeline will at first serve a basic purpose in shuttling and preparing epochs to be used within a EEGNet model (available here: https://raw.githubusercontent.com/vlawhern/arl-eegmodels/master/EEGModels.py and
https://github.com/vlawhern/arl-eegmodels/tree/master)

Referenced as below:

@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}

EEGNet is Implemented using TensorFlow/Keras, but further experimentation may take place in utilizing PyTorch.
