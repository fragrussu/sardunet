# SARDU-Net: tutorials

The python class `sardunet_v1` included in the [`sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/sardunet.py) file defines the architecture of SARDU-Net. It contains the following methods:
* `__init__()`: the constructor, to define the hidden layers of a SARDU-Net;
* `selector()`: to pass data through the *selector* sub-network of a SARDU-Net;
* `predictor()`: to pass data through the *predictor* sub-network of a SARDU-Net;
* `forward()`: to pass data through the entire SARDU-Net.

Each method has a detailed *help*, which you can print from python this way:
```
import sardunet
help(sardunet.sardunet_v1)
```

Additionaly, a number of command line tools are provided to help you train and use SARDU-Net objects: 

* [`extractvoxels_sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/extractvoxels_sardunet.py) to extract measurements from qMRI scans storerd in NIFTI format as required to train a SARDU-Net;

* [`train_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/train_sardunet_v1.py) to train a SARDU-Net;

* [`downsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) to downsample a qMRI scan keeping measurements selected by SARDU-Net.=;

* [`upsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) to upsamples a qMRI scan containing the measurements selected by SARDU-Net to a richer qMRI protocol.


All tools listed above have a detailed *help* manual, which you can print by simply typing in your terminal `python </PATH/TO/TOOL> --help` (for example, `python train_sardunet_v1.py --help`). 

Additionally, these simple tutorials show the tools in action:  

* [**Tutorial 1**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial1.md) shows how to extract voxels for SARDU-Net training; 

* [**Tutorial 2**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial2.md) shows how to train a SARDU-Net, choosing the learning options and accessing qMRI sub-protocols selected by the trained SARDU-Net; 

* [**Tutorial 3**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial3.md) shows how to use a trained SARDU-Net to downsample or upsample a qMRI scan.
