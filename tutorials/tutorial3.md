# Tutorial 3: downsample/upsample qMRI with SARDU-Net
A trained SARDU-Net has learnt a mapping between a short, clinically viable qMRI protocol and a richer, densely-sampled qMRI scan, in order to point you towards informative measurements. You could stop here, and make use of this information to set up  a shorter qMRI scan including only the measurements selected by SARDU-Net. However, in some cases you could try to exploit this mapping to upsample qMRI scans acquired with the shorter protocol in question, and estimate how they would look like if the full protocol had been acquired. 

**Big disclaimer**: SARDU-Net upsampling must be limited to a pure computational MRI research context and ***no conclusions on disease from the upsampled images should be drawn***. As any deep learning method, there are potential issues in doing so: *What if I blur a lesion>?* *What if I add artifactual features not present in my data?*. Extensive validation of the upsampled data is still required before SARDU-Net upsampling could be use to inform clinical decisions based on the inspection of the upsampled images.


Below, it is assumed that you have run [tutorial 1](https://github.com/fragrussu/sardunet/blob/master/tutorials/tutorial1.md) and [tutorial 2](https://github.com/fragrussu/sardunet/blob/master/tutorials/tutorial2.md) and you have trained a SARDU-Net. It is also assumed that the variable `SARDULIB` stores the location of the SARDU-Net tools in your file system (for example: `SARDULIB="/Users/myname/lib/python/sardunet/ainas"`), and that the path to the directory storing the trained SARDU-Net is stored in variable `TRAINDIR` (for example: `TRAINDIR=sardutrain/sarduout_nnsel16-12-16_psel0.2_dsel8_nnpred16-12-16_ppred0.2_noepoch500_lr0.001_mbatch1000_seed257891_lossL2_prct99.0_small1e-06`).


## a) Downsample a qMRI scan
Let's say that file `testscan.nii` contains a scan that could be used as test set, i.e. a scan that was not included in training/validation sets when training a SARDU-Net. Let's also assume that this scan was acquired with the full protocol used to train SARDU-Net. You can extract the sub-protocol selected by SARDU-Net and save it to a new `testscan_down.nii` file with the `downsample_sardunet_v1.py`. Simply run:
```
python $SARDULIB/downsample_sardunet_v1.py testscan.nii testscan_down.nii $TRAINDIR/nnet_lossvalmin.bin
```
and you should see something like:

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/sardudown.png" width="512">


Above, the SARDU-Net `$TRAINDIR/nnet_lossvalmin.bin` was used (it stores the indices of the selected measurements). By default `testscan_down.nii` will be saved using `FLOAT32` precision; use option `--bits 64` if you prefer `FLOAT64`.


## b) Upsample a qMRI scan
Finally, you can use a trained SARDU-Net to estimate how a richer protocol would look like if you data acquired according to the sub-protocol selected by SARDU-Net. For instance, we can try to upsample back the sub-protocol extracted above (`testscan_down.nii`) and get an estimate of the fully-sampled scan (`testscan_down_up.nii`, an estimate of `testscan.nii` obtained directly from `testscan_down.nii`). You can do this with the `upsample_sardunet_v1.py` tool: if you run
```
python $SARDULIB/upsample_sardunet_v1.py testscan_down.nii testscan_down_up.nii $TRAINDIR/nnet_lossvalmin.bin $TRAINDIR/max_val.bin $TRAINDIR/min_val.bin
```
you should see something like

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/sarduup.png" width="512">

Above, the SARDU-Net `$TRAINDIR/nnet_lossvalmin.bin` was used (it stores the indices of the selected measurements), and data normalisation is performed before passing data through the network (this is the reason why we need to specify normalistion parameters `$TRAINDIR/max_val.bin` and `$TRAINDIR/min_val.bin`). Similarly to `downsample_sardunet_v1`, by default `downsample_sardunet_v1.py` saves its prediction using `FLOAT32` precision: use option `--bits 64` if you prefer `FLOAT64` instead.

Below you can find an example of what you can expect to get when upsampling a qMRI protocol with SARDU-Net. The figure shows prostate [diffusion-relaxation images](https://doi.org/10.1101/2020.05.26.116491) obtained at 3Tesla, and illustrates an upsampling from a sub-protocol of 9 contrasts back to the original measurement space of 16 contrasts. *Note that SARDU-Net provides a smooth estimate even for those measurements that were actually acquired as part of a sub-protocol!*


<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/sarduimages.png" width="1024">


