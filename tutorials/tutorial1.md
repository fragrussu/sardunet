# Tutorial 1: extract voxels for SARDU-Net training
SARDU-Net is designed to identify informative sub-protocols from a set of rich, densely-sampled qMRI scans, given a budget in terms of number of mesaurements. The first thing you will need to do to train a SARDU-Net is to extract the qMRI measurements from voxels of interest within the pilot training scans. 


Let's imagine that you have, for instance, pilot densely-sampled qMRI scans from *N* subjects (*N = 1* would work perfectly fine too). These could encompass any combination of qMRI contrasts acquired in any anatomical district - say, for instance, multi b-shell diffusion-weighted scans performed at several different echo times. 

Below it is asssumed that you will be using SARDU-Net in a [bash shell](https://www.gnu.org/software/bash). Also, it is assumed that the variable `SARDULIB` stores the location of the [SARDU-Net tools](https://github.com/fragrussu/sardunet/tree/master/ainas) in your file system (for example: `SARDULIB="/Users/myname/lib/python/sardunet/ainas"`).  


## a) Prepare your training data
* Create a training folder - name it, for instance, `sardutrain`.


* Put your scans inside `sardutrain`. Make sure your scans are in [NIFTI1](https://nifti.nimh.nih.gov/nifti-1) format (I use [dcm2niix](https://github.com/rordenlab/dcm2niix) to convert DICOMs to NIFTI).


* Make sure you name your scans as `scan_1.nii`, ..., `scan_N.nii`. Each scan should be a 4D NIFTI, with different qMRI measurements arranged along the 4th dimension. All scans should have same number of qMRI measurements, and in the same order.

* If you have binary masks of the organ/tissue of interest, include them in the same folder, making sure to name them as `mask_1.nii`, ..., `mask_N.nii` (obviously `mask_1.nii` corresponds to `scan_1.nii`, `mask_2.nii` corresponds to `scan_2.nii`, etc).

* MRI signal intensities from the different scans should be broadly comparable. While it is understood that each scan will have its own rx/tx smooth bias field, wild differences in signals intensities (for example: signals in `scan_1.nii` vary between 5.0 and 9.0, while signals in `scan_2.nii` vary between 1250.0 and 7000.0) would affect SARDU-Net performance. A simple way to avoid such wild intensity variations is to divide each scans by its median signal intensity. This could be done easily with [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki): assuming that you have, for instance, 3 scans (so *N = 3*), you can run something like:
```
for ii in `seq 1 3`
do
	normsig=`fslstats "scan_"$ii".nii" -p 50`
	fslmaths "scan_"$ii".nii" -div $normsig "scan_"$ii".nii"
done
```
or
```
for ii in `seq 1 3`
do
	normsig=`fslstats "scan_"$ii".nii" -k "mask_"$ii".nii" -p 50`
	fslmaths "scan_"$ii".nii" -div $normsig "scan_"$ii".nii"
done
```
if you have masks.

## b) Extract voxels
You are now ready to extract qMRI measurements from the voxels in your training set. Voxels will be split randomly into an actual training set and a validation set. You can control the ratio between the number of voxels kept of training and the number of voxels allocated as validation set).

* Navigate to the `sardutrain` directory, where scans `scan_1.nii`, ..., `scan_N.nii` are stored.

* Extract voxels. If you do not have masks, type:
```
python $SARDULIB/extractvoxels_sardunet.py . datatrain.bin dataval.bin
```
or use the `--usemask 1` flag if you have masks:
```
python $SARDULIB/extractvoxels_sardunet.py . datatrain.bin dataval.bin --usemask 1
```
You should see something like:

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/extractverbose.png" width="48">

![extractverbose](https://github.com/fragrussu/sardunet/blob/master/tutorials/extractverbose.png)

* This should have extracted in the current folder training and validation sets. They should have been saved as `datatrain.bin` (training set) and `dataval.bin` (validation set). These are binary files that store MRI signals as 2D matrices (rows: different voxels; columns: different qMRI measurements). Here an example of the content `datatrain.bin` of from some diffusion-relaxation imaging data:

![sigmat](https://github.com/fragrussu/sardunet/blob/master/tutorials/sigmat.png)

* By default, 80% of voxels are kept in training set, while 20% are used as validation set. You can change this ratio with the `--valratio` option (default is 0.2, i.e. 20% of voxels to be used for validation). 

* Finally, options `--matlab_train` and `--matlab_val` allow you to save copies of the training and/or validation sets in [Matlab](https://mathworks.com) format.

## c) Ready to train a SARDU-Net!
You are now ready to train a SARDU-Net. [Tutorial 2](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial2.md) shows you how to do this.
