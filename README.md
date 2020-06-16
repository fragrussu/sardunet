# SARDU-Net: overview
"*Select and retrieve via direct upsampling*" network (SARDU-Net) enables model-free, data-driven quantitative MRI (qMRI) protocol design.

SARDU-Net selects subsets of informative qMRI measurements within lengthy pilot scans. The algorithm consists of two deep neural networks (DNNs) that are trained jointly end-to-end: a *selector* identifies a subset of input qMRI measurements, while a *predictor* estimates the fully-sampled signals from such a subset. 

The joint optimisation of *selector* and *predictor* DNNs enables the selection of informative sub-protocols of fixed size (i.e. shorter, clinically feasible) from densely-sampled qMRI signals (i.e. longer, difficult to implement in real clinical scenarios). 

In the SARDU-Net framework, the densely-sampled qMRI measurements would typically come from a few rich pilot qMRI scans that are performed any way when setting up new MRI studies for quality control. These could include data from patients and data augmentation techniques could be used to increase signal examples from under-represented tissue types.

The code will be made avaiable in the coming days... Apologies for the delay!


# Dependencies
To use SARDU-Net you need a Python 3 distribution such as [Anaconda](http://www.anaconda.com/distribution). Additionally, you need the following third party modules/packages:
* [NumPy](http://numpy.org)
* [Nibabel](http://nipy.org/nibabel)
* [SciPy](http://www.scipy.org)
* [PyTorch](http://pytorch.org/)


# Download 
Getting SARDU-Net is extremely easy: cloning this repository is all you need to do. The tools would be ready for you to run.

If you use Linux or MacOS:

1. Open a terminal;
2. Navigate to your destination folder;
3. Clone SARDU-Net:
```
git clone https://github.com/fragrussu/sardunet.git 
```
4. SARDU-Net is ready for you in `./sardunet`. SARDU-Net tools are available here: 
```
./sardunet/ainas
```
(*ainas* means *tools* in Sardinian language), while a number of tutorials are available here:
```
./sardunet/tutorials
```
5. You should now be able to use the code. Try to print the manual of a script, for instance of `extractvoxels_sardunet.py`, to make sure this is really the case:
```
python ./sardunet/tools/extractvoxels_sardunet.py --help
```

# Citation
If you use SARDU-Net, please remember to cite our work:

"SARDU-Net: a new method for model-free, data-driven experiment design in quantitative MRI". Grussu F, Blumberg SB, Battiston M, Ianuș A, Singh S, Gong F, Whitaker H, Atkninson D, Gandini Wheeler-Kingshott CAM, Punwani S, Panagiotaki E, Mertzanidou T and Alexander DC. Proceedings of the 2020 virtual annual meeting of the International Society for Magnetic Resonance in Medicine (ISMRM). 

# License
SARDU-Net is distributed under the BSD 2-Clause License, Copyright (c) 2020 University College London. All rights reserved.
Link to license [here](http://github.com/fragrussu/sardunet/blob/master/LICENSE).

# Acknowledgements
The development of SARDU-Net was funded by the Engineering and Physical Sciences Research Council (EPSRC EP/R006032/1). This project has also received funding under the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 634541 and No. 666992. Funding has also been received from: EPSRC (M020533/1, G007748, I027084, N018702); Rosetrees Trust (UK, funding FG); Prostate Cancer UK Targeted Call 2014 (Translational Research St.2, project reference PG14-018-TR2); Spinal Research (UK), Wings for Life (Austria), Craig H. Neilsen Foundation (USA) for jointly funding the INSPIRED study; Wings for Life (#169111); UK Multiple Sclerosis Society (grants 892/08 and 77/2017); the Department of Health's National Institute for HealthResearch (NIHR) Biomedical Research Centres and UCLH NIHR Biomedical Research Centre; Champalimaud Centre for the Unknown, Lisbon (Portugal); European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101003390.
