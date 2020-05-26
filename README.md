# SARDU-Net: overview
"Select and retrieve via direct upsampling" network (SARDU-Net) enables model-free, data-driven quantitative MRI (qMRI) protocol design.

SARDU-Net selects subsets of informative qMRI measurements within lengthy pilot scans. The algorithm consists of two deep neural networks (DNNs) that are trained jointly end-to-end: a *selector* identifies a subset of input qMRI measurements, while a *predictor* estimates the fully-sampled signals from such a subset. 

The joint optimisation of *selector* and *predictor* enables the selection of informative sub-protocols of fixed size (i.e. shorter, clinically feasible) from densely-sampled qMRI signals (i.e. longer, difficult to implement in real clinical scenarios). 

In the foreseen application of SARDU-Net, the densely-sampled qMRI measurements would typically come from a few rich pilot qMRI scans that are performed any way when setting up new MRI studies for quality control. These could include data from patients and data augmentation techniques could be used to increase signal examples from under-represented tissue types.

# Dependencies
To use SARDU-Net you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [NumPy](http://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [SciPy](http://www.scipy.org)
* [PyTorch](http://pytorch.org/)


# Download 
Getting *sardunet* is extremely easy: cloning this repository is all you need to do. The tools would be ready for you to run.

If you use Linux or MacOS:

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone MRItools:
```
git clone https://github.com/fragrussu/sardunet.git 
```
4. MRItools is ready for you in `./sardunet` with the tools available here: 
```
./sardunet/tools
```
5. You should now be able to use the code. Try to print the manual of a script, for instance of `extractvoxels_sardunet.py`, to make sure this is really the case:
```
python ./sardunet/tools/extractvoxels_sardunet.py --help
