# Tutorial 2: train a SARDU-Net
This tutorial demonstrates how to train a SARDU-Net given a training and validation sets of qMRI measurements. 

It is assumed that qMRI measurements have been extracted to binary files `datatrain.bin` and `dataval.bin` in a folder named `sardutrain`, as described in [Tutorial 1](https://github.com/fragrussu/sardunet/blob/master/tutorials/tutorial1.md). It is also assumed that the variable `SARDULIB` stores the location of the [SARDU-Net tools](https://github.com/fragrussu/sardunet/tree/master/ainas) in your file system (for example: `SARDULIB="/Users/myname/lib/python/sardunet/ainas"`).

At present, SARDU-Net can be trained only on CPU. We plan to release a GPU implementation soon.

## a) Default training
Training a SARDU-Net is extremely easy: all you have to do is to specify training and validation sets (here `datatrain.bin` and `dataval.bin`, which contain 16 diffusion-relaxation measurements), as well as a root string for the output folder (say, for example, `sarduout`):
```
python $SARDULIB/train_sardunet_v1.py datatrain.bin dataval.bin sarduout
```

This should print something along these lines:

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/sardutrain.png" width="1024">

By default, a SARDU-Net searching sub-protocols containing half the number of the input qMRI measurements is trained, i.e. here 8 out 16 measurements (if you need to select a different number of measurements, use option ` --dsel`). Also, by default training is performed for 500 epochs with dropout regularisation of 0.2, learning rate of 0.001, and using mini-batches of 1000 voxels to minimise an L2 loss. Two layers are used for both *selector* and *predictor* neural networks; signals smaller than 10<sup>-6</sup> are clamped to this number and then all signals normalised to the 99% percentile of the signal distribution.  

Training results will be stored in an output directory whose name contains all salient information regarding training, which are appended to the root string specified on command line. In this example, the output directory will be called `sarduout_nnsel16-12-16_psel0.2_dsel8_nnpred16-12-16_ppred0.2_noepoch500_lr0.001_mbatch1000_seed257891_lossL2_prct99.0_small1e-06`. Its content will be:

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/sarduout.png" width="1024">

The most important output is perhaps `nnet_lossvalmin_measidx.txt`, storing the indices of the selected qMRI measurements (starting from 0!). For example,
```
0
2
4
5
7
8
11
12
```
means that measurement 0 was selected, measurement 1 discarded, measurement 2 selected, measurement 3 discarded, etc ...


The meaning of all other output files is described briefly below. 
* `losstrain.bin`: training loss values as a 2D matrix (epoch x minibatch) 
* `lossval.bin`: validation loss values as a 1D array (1 value per epoch)
* `max_val.bin`: maximum signal value for normalisation
* `min_val.bin`: minimum signal value for normalisation
* `meas_idx.bin`: measurement selected after each network update as a 3D array (epoch x minibatch x number of selected measurements)
* `meas_weight.bin`: measurement score after each network update as a 3D array (epoch x minibatch x number of input measurements)
* `nnet_epoch0.bin`: SARDU-Net at initialisation
* `nnet_epoch0.pth`: SARDU-Net at initialisation (pytorch `.pth` file extension) 
* `nnet_lossvalmin.bin`: SARDU-Net at epoch minimising the validation loss
* `nnet_lossvalmin.pth`: SARDU-Net at epoch minimising the validation loss (pytorch `.pth` file extension)
* `nnet_lossvalmin_measidx.txt`: indices of selected measurements at the epoch minimising the validation loss (indices start from 0!)
* `nnet_lossvalmin_sigval.bin`: prediction of validation signals at epoch minimising the validation loss
* `nnet_lossvalmin.info`: text file with information regarding the epoch minimising the validation loss
* `nnet_epoch500.bin`: SARDU-Net at last epoch
* `nnet_epoch500.pth`: SARDU-Net at last epoch (pytorch `.pth` file extension)
* `nnet_epoch500_measidx.txt`: indices of selected measurements at the last epoch (indices start from 0!)

You can load all binary files (`.bin`) into memory with [pickle module](https://docs.python.org/3/library/pickle.html) in python. For example, in your [Jupyter notebook](https://jupyter.org) or in your python interpreter prompt you can type something like:
```
>>> import numpy as np
>>> import pickle as pk
>>> h = open('losstrain.bin','rb')
>>> loss = pk.load(h) 
>>> h.close()
>>> loss.shape
(500,6)
```
showing that the file `losstrain.bin` in this case contains information from 500 epochs, each split in 6 mini-batches.


## b) Explore different training options
There are several training options you can tune, as for example: learning rate (` --lrate`), mini-batch size (`--mbatch`), number of epochs (`--noepoch`), dropout regularisation (`--psel` and `--ppred`). Also, you can try different network architectures (e.g. number and depth of hidden layers of both selector and predictor, `--nnsel` and `--nnpred`).



A practical way to find an acceptable SARDU-Net architecture and good learning options is to try several different combinations, in a grid search fashion. The combination providing the lowest validation loss could then be chosen for subsequent experiments. 


Finally, you can also train a SARDU-Net minimising a L1 loss (option `--lossnorm`), using different signal normalisation strategies (options `--prct` and `--small`) and using a specific seed number to initialise the network parameters (option `--seed`).


## b) Use a trained SARDU-Net to upsample qMRI experiments
In most applications finding informative sub-protocols from rich pilot scans should be enough. You could now pick the indices of the selected measurements and go to the scanner room to set up your new qMRI experiment, which should now be considerably shorter than the previous one used to acquire the pilot training data.


However, do not forget that SARDU-Net has learnt a mapping between a short, clinically viable qMRI protocol and a richer, densely-sampled qMRI scan. You could try to exploit this mapping to upsample qMRI scans acquired with the shorter protocol in question, and estimate how they would look like if the full protocol had been acquired. This is exactly the idea behind the 2019 MICCAI CDMRI workshop challenge known as ['MUDI'](http://cmic.cs.ucl.ac.uk/cdmri19/challenge.html), which SARDU-Net won in October 2019. 


[Tutorial 3](https://github.com/fragrussu/sardunet/blob/master/tutorials/tutorial3.md) will show you how to downsample/upsample qMRI experiments with a trained SARDU-Net.
