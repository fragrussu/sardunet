# SARDU-Net: tutorials

The python class `sardunet_v1` included in the [`sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/sardunet.py) file defines the architecture of SARDU-Net. It contains the following methods:
* `__init__()`: the constructor, to define the hidden layers of a SARDU-Net;
* `selector()`: to pass data through the *selector* sub-network of a SARDU-Net;
* `predictor()`: to pass data through the *predictor* sub-network of a SARDU-Net;
* `forward()`: to pass data through the entire SARDU-Net.

Each method has a detailed *help manual*. From a [Jupyter notebook](https://jupyter.org) or in your python interpreter prompt, you can check the manual by typing:
```
>>> import sardunet
>>> help(sardunet.sardunet_v1)
```

Additionaly, a number of command line tools are provided to help you train and use SARDU-Net objects: 

* [`extractvoxels_sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/extractvoxels_sardunet.py) to extract measurements from qMRI scans as required to train a SARDU-Net;

* [`train_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/train_sardunet_v1.py) to train a SARDU-Net;

* [`downsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) to downsample a qMRI scan keeping measurements selected by SARDU-Net;

* [`upsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) to upsample a qMRI scan from a SARDU-Net sub-protocol to a richer qMRI protocol.


All tools listed above also have a detailed *help manual*, which you can print by simply typing in your terminal `python </PATH/TO/TOOL> --help` (for example, `python train_sardunet_v1.py --help`). These tutorials show SARDU-Net tools at work:  

* [**Tutorial 1**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial1.md) shows how to extract voxels for SARDU-Net training; 

* [**Tutorial 2**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial2.md) shows how to train a SARDU-Net, choosing the learning options and accessing qMRI sub-protocols selected by the trained SARDU-Net; 

* [**Tutorial 3**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial3.md) shows how to use a trained SARDU-Net to downsample or upsample a qMRI scan.
