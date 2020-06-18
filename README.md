# SARDU-Net
"***Select and retrieve via direct upsampling***" network (SARDU-Net) enables model-free, data-driven quantitative MRI (qMRI) protocol design.

SARDU-Net selects subsets of informative qMRI measurements within lengthy pilot scans. The algorithm consists of two deep neural networks (DNNs) that are trained jointly end-to-end: a ***selector*** identifies a subset of input qMRI measurements, while a ***predictor*** estimates the fully-sampled signals from such a subset. 

The joint optimisation of *selector* and *predictor* DNNs enables the selection of informative sub-protocols of fixed size (i.e. shorter, clinically feasible) from densely-sampled qMRI signals (i.e. longer, difficult to implement in real clinical scenarios). 

In SARDU-Net, the densely-sampled qMRI measurements would typically come from a few rich pilot qMRI scans that are performed any way when setting up new MRI studies for quality control. These could include data from patients and data augmentation techniques could be used to increase signal examples from under-represented tissue types.

The figure below shows a schematic representation of the general idea behind the SARDU-Net framework.

<img src="https://github.com/fragrussu/sardunet/blob/master/tutorials/architecture.png" width="512">


## Dependencies
To use SARDU-Net you will need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you will also need:
* [NumPy](http://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [SciPy](http://www.scipy.org)
* [PyTorch](http://pytorch.org/)


## Download 
Getting SARDU-Net is extremely easy: cloning this repository is all you need to do. The tools would be ready for you to run.

If you use Linux or MacOS:

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone SARDU-Net:
```
git clone https://github.com/fragrussu/sardunet.git 
```
4. SARDU-Net tools (i.e. a python class and command-line tools to train and use objects of that class) are now available in the [`./sardunet/ainas`](https://github.com/fragrussu/sardunet/tree/master/ainas) folder (*ainas* means *tools* in [**Sardinian language**](http://sc.wikipedia.org/wiki/Limba_sarda)), while [`./sardunet/tutorials`](https://github.com/fragrussu/sardunet/tree/master/tutorials) contains a number of tutorials. 
5. You should now be able to use the code. Try to print the manual of a tool, for instance of `train_sardunet_v1.py`, to make sure this is really the case:
```
python ./sardunet/ainas/train_sardunet_v1.py --help
```

## SARDU-Net class and tutorials
SARDU-Net is implemented in the `sardunet_v1` python class, defined in the [`sardunet.py`](https://github.com/fragrussu/sardunet/blob/master/ainas/sardunet.py) file. Details on `sardunet_v1` methods can be found [here](https://github.com/fragrussu/sardunet/blob/master/tutorials/README.md). 

Command line tools are also provided to help you train and use `sardunet_v1` objects. These tutorials demonstrate how to use the tools:  

* [**Tutorial 1**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial1.md) shows how to extract voxels for SARDU-Net training; 

* [**Tutorial 2**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial2.md) shows how to train a SARDU-Net, choosing the learning options and accessing qMRI sub-protocols selected by the trained SARDU-Net; 

* [**Tutorial 3**](https://github.com/fragrussu/sardunet/tree/master/tutorials/tutorial3.md) shows how to use a trained SARDU-Net to downsample or upsample a qMRI scan.

## Citation
If you use SARDU-Net, please remember to cite our work:

"*Select and retrieve via direct upsampling* network (SARDU-Net): a data-driven, model-free, deep learning approach for quantitative MRI protocol design". Grussu F, Blumberg SB, Battiston M, Kakkar LS, Lin H, Ianuș A, Schneider T, Singh S, Bourne R, Punwani S, Atkinson D, Gandini Wheeler-Kingshott CAM, Panagiotaki E, Mertzanidou T and Alexander DC. biorxiv 2020, DOI: [10.1101/2020.05.26.116491](https://doi.org/10.1101/2020.05.26.116491). 

"SARDU-Net: a new method for model-free, data-driven experiment design in quantitative MRI". Grussu F, Blumberg SB, Battiston M, Ianuș A, Singh S, Gong F, Whitaker H, Atkinson D, Gandini Wheeler-Kingshott CAM, Punwani S, Panagiotaki E, Mertzanidou T and Alexander DC. Proceedings of the 2020 virtual annual meeting of the International Society for Magnetic Resonance in Medicine (ISMRM). 

## License
SARDU-Net is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/sardunet/blob/master/LICENSE).

## Acknowledgements
The development of SARDU-Net was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.
