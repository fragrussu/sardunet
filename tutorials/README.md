# SARDU-Net: tutorials

The python class `sardunet_v1` included in the [`sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/sardunet.py) file defines the architecture of SARDU-Net. The following command line tools are also provided: 

* [`extractvoxels_sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/extractvoxels_sardunet.py) extracts measurements from qMRI scans storerd in NIFTI format to train a SARDU-Net;

* [`train_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/train_sardunet_v1.py) trains a SARDU-Net;

* [`upsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) upsamples a qMRI scan containing the measurements selected by SARDU-Net to a richer qMRI protocols;

* [`downsample_sardunet_v1.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/downsample_sardunet_v1.py) downsamples a qMRI scan keeping measurements selected by SARDU-Net.


All tools listed above contain a detailed *help* manual, which you can print by simply typing in your terminal `python </PATH/TO/TOOL> --help`. Additionally, these simple tutorials show the tools in action:  

* [**Tutorial 1**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial1.md) shows how to extract voxels for SARDU-Net training. 

* [**Tutorial 2**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial2.md) shows how to train a SARDU-Net, choosing the learning options and accessing qMRI sub-protocols selected by the trained SARDU-Net; 

* [**Tutorial 3**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial3.md) shows how to use a trained SARDU-Net to downsample or upsample a qMRI scan.
