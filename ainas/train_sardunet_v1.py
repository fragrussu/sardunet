# Author: Francesco Grussu, University College London
#		   <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>
#
# Code released under BSD Two-Clause license
#
# Copyright (c) 2020 University College London. 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

### Load libraries
import argparse, os, sys
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch import autograd
import pickle as pk
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import sardunet


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a "Select and Retrieve for Direct Upsampling" network (SARDU-net). A SARDU-net finds the indices of the informative measurements within a rich multi-contrast MRI data set, and learns a mapping from a subsampled protocol containing such informative measurements and a richer protocol. SARDU-net version v1 works voxel-by-voxel and implements a two-step procedure, with two deep neural networs (DNNs) working sequentially: the first DNN selects the optimal subset of measurements, while the second DNN network retrieves the fully sampled set of measurements from the given subset. The process is optimised jointly, so that the selected measurements are those that enable the best prediction of the fully sampled signal. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('data_train', help='path to a pickle binary file storing the input training data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('data_val', help='path to a pickle binary file storing the validation data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('out_base', help='base name of output directory (a string built with the network parameters will be added to the base). The output directory will contain the following output files: * losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); * lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); * max_val.bin, pickle binary storing the upper bound for signal normalisation as a variable; * min_val.bin, pickle binary storing the lower bound for signal normalisation as a variable; * meas_idx.bin, pickle binary storing the indices of measurements selected by the SARDU-net selector during training as a numpy matrix (shape: epoch x batch x measurement); * meas_weight.bin, pickle binary storing the measurement weights during training as a numpy matrix (shape: epoch x batch x measurement); * nnet_lossvalmin_measidx.txt, text file storing the indices of the measurements selected by SARDU-net at the best epoch (minimum validation loss) as a text file; * nnet_epoch0.bin, pickle binary storing the SARDU-net at initialisation; * nnet_epoch0.pth, Pytorch binary storing the SARDU-net at initialisation; * nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the SARDU-net at the final epoch; * nnet_epoch<FINAL_EPOCH>.pth, Pytorch binary storing the SARDU-net at the final epoch; * nnet_epoch<FINAL_EPOCH>_measidx.txt, text file storing the indices of the measurements selected by SARDU-net at the last epoch as a text file; * nnet_lossvalmin.bin, pickle binary storing the trained SARDU-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.pth, Pytorch binary storing the trained SARDU-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin_sigval.bin, prediction of the validation signals (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss')
	parser.add_argument('--ttar', metavar='<path>', help='path to a pickle binary file storing a second training data set as a numpy matrix (rows: voxels; columns: measurements; this will be the target of the entire prediction)')
	parser.add_argument('--vtar', metavar='<path>', help='path to a pickle binary file storing a second validation data set as a numpy matrix (rows: voxels; columns: measurements; this will be the target of the entire prediction)')
	parser.add_argument('--fte', metavar='<value>',  help='index of first echo time (TE) if data contains multi-echo data to be analysed in block (indices start from 0; if set, SARDU-net will optimise the maximum TE and will keep all the TEs smaller than the optimsed one). If this parameter is set, then it is compulsory to set parameter nte too')
	parser.add_argument('--nte', metavar='<value>',  help='number of echo times (TEs) if data contains multi-echo data to be analysed in block (indices start from 0; if set, SARDU-net will optimise the maximum TE and will keep all the TEs smaller than the optimsed one). If parameter fte is set, it is compulsory to set this parameter')
	parser.add_argument('--nnsel', metavar='<list>', help='array storing the number of hidden neurons in the selector, separated by hyphens (example: 30-15-30). Default: Nin-(Nin+dsel)/2-Nin, where Nin is the number of input measurements and dsel the number of measurements to be selected by the selector')
	parser.add_argument('--psel', metavar='<value>', default='0.2', help='dropout probability in each layer of the selector. Default: 0.2')
	parser.add_argument('--dsel', metavar='<value>', help='number of measurements to be selected by the selector. Default: Nin/2, i.e. half of the input measurements')
	parser.add_argument('--nnpred', metavar='<list>', help='array storing the number of hidden neurons in the predictor, separated by hyphens (example: 30-15-30). Note: the first/last layers of the predictor must have the same number of neurons as the last/first layers of the selector. Default: Nin-(dsel+Nout)/2-Nout, where Nin is the number of input measurements, dsel is the number of measurements selected by the selector and Nout is the number of output measurements')
	parser.add_argument('--ppred', metavar='<value>', default='0.2', help='dropout probability in each layer of the predictor. Default: 0.2')
	parser.add_argument('--noepoch', metavar='<value>', default='500', help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', default='1000', help='number of voxels in each mini-batch. Default: 1000')
	parser.add_argument('--seed', metavar='<value>', default='257891', help='integer used as a seed for Numpy and PyTorch random number generators. Default: 257891')
	parser.add_argument('--nwork', metavar='<value>', default='0', help='number of workers for data loader. Default: 0')
	parser.add_argument('--lossnorm', metavar='<value>', default='2', help='optimse L2 or L1 loss (2 for L2, 1 for L1). Default: 2 (L2 loss)')
	parser.add_argument('--prct', metavar='<value>', default='99', help='percentile used for log-signal normalisation. Default: 99 (99th percentile)')
	parser.add_argument('--small', metavar='<value>', default='1e-6', help='minimum signal level allowed. Default: 1e-6 (everything smaller than this number will be clamped to this value)')
	args = parser.parse_args()

	### Get some of the inputs
	psel = float(args.psel)
	ppred = float(args.ppred)
	noepoch = int(args.noepoch)
	lrate = float(args.lrate)
	mbatch = int(args.mbatch)
	seed = int(args.seed)
	nwork = int(args.nwork)
	lossflag = int(args.lossnorm)
	prctsig = float(args.prct)
	smallsig = float(args.small)
	if args.nte is not None:
		nte = int(args.nte)
		if args.fte is None:
			raise RuntimeError('parameter nte is set (you set {}) but parameter fte is not!'.format(args.nte))
	if args.fte is not None:
		fte = int(args.fte)
		if args.nte is None:
			raise RuntimeError('parameter fte is set (you set {}) but parameter nte is not!'.format(args.fte))

	### Check for some obvious mistakes	
	if args.ttar is not None:
		if args.vtar is None:
			raise RuntimeError('you have specified target training data with option --ttar but you have not specified target validation data with option --vtar!')

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                    TRAIN A S.A.R.D.U.-NET                          ')
	print('********************************************************************')
	print('')
	print('** Input training data: {}'.format(args.data_train))
	print('** Input validation data: {}'.format(args.data_val))
	if args.ttar is not None:
		print('** Input target training data: {}'.format(args.ttar))
	if args.vtar is not None:
		print('** Input target validation data: {}'.format(args.vtar))


	### Load training data
	fh = open(args.data_train,'rb')
	datatrain = np.float32(pk.load(fh))
	nmeas_train = datatrain.shape[1]
	nmeas_out = nmeas_train 
	fh.close()

	### Load validation data
	fh = open(args.data_val,'rb')
	dataval = np.float32(pk.load(fh))
	fh.close()

	### Load target data if requested
	if args.ttar is not None:

		fh = open(args.ttar,'rb')
		datattar = np.float32(pk.load(fh))
		nmeas_ttar = datattar.shape[1]
		nmeas_out = nmeas_ttar
		fh.close()

		fh = open(args.vtar,'rb')
		datavtar = np.float32(pk.load(fh))
		fh.close()


	# Get parameters for neural network
	if args.dsel is None:
		dsel = int(nmeas_train/2)
	else:
		dsel = int(args.dsel)

	if args.nnsel is None:
		nhidden_sel = np.array([int(nmeas_train) , int(0.5*float(dsel)+0.5*float(nmeas_train)) , int(nmeas_train)])
		nnsel_str = '{}-{}-{}'.format( int(nmeas_train) , int(0.5*float(dsel)+0.5*float(nmeas_train)) , int(nmeas_train)  )

	else:
		nhidden_sel = (args.nnsel).split('-')
		nhidden_sel = np.array( list(map( int,nhidden_sel )) )
		nnsel_str = args.nnsel

	if args.nnpred is None:
		nhidden_pred = np.array([int(nmeas_train) , int(0.5*float(dsel)+0.5*float(nmeas_out)) , int(nmeas_out)])
		nnpred_str = '{}-{}-{}'.format( int(nmeas_train) , int(0.5*float(dsel)+0.5*float(nmeas_out)) , int(nmeas_out)  )

	else:
		nhidden_pred = (args.nnpred).split('-')
		nhidden_pred = np.array( list(map( int,nhidden_pred )) )
		nnpred_str = args.nnpred


	### Check for some other obvious mistakes
	if(dsel>=nhidden_sel[-1]):
		raise RuntimeError('parameter dsel (you set {}) should be less than the output number of selector neurons (you set {})!'.format(dsel,nhidden_sel[-1]))

	if(nhidden_sel[0]!=nhidden_sel[-1]):
		raise RuntimeError('the number of input and output neurons for the predictor (you set {} and {}) must be the same!'.format(nhidden_sel[0],nhidden_sel[-1]))	

	if(nhidden_sel[-1]!=nhidden_pred[0]):
		raise RuntimeError('the number of output neurons for the selector (you set {}) must be the same as the input neurons of the predictor (you set {})!'.format(nhidden_sel[-1],nhidden_pred[0]))


	### Create output base name
	out_base_dir = '{}_nnsel{}_psel{}_dsel{}_nnpred{}_ppred{}_noepoch{}_lr{}_mbatch{}_seed{}_lossL{}_prct{}_small{}'.format(args.out_base,nnsel_str,psel,dsel,nnpred_str,ppred,noepoch,lrate,mbatch,seed,lossflag,prctsig,smallsig)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir)

	### Print some more information
	print('** Output directory: {}'.format(out_base_dir))
	print('')
	print('')
	print('PARAMETERS')
	print('')
	print('** Hidden neurons of selector: {}'.format(nhidden_sel))
	print('** Dropout selector: {}'.format(psel))
	print('** Number of measurements to be kept by the selector: {}'.format(dsel))
	print('** Hidden neurons of predictor: {}'.format(nhidden_pred))
	print('** Dropout predictor: {}'.format(ppred))
	print('** Number of epochs: {}'.format(noepoch))
	print('** Learning rate: {}'.format(lrate))
	print('** Number of voxels in a mini-batch: {}'.format(mbatch))
	print('** Seed: {}'.format(seed))
	print('** Number of workers for data loader: {}'.format(nwork))
	print('** Loss: L{}'.format(lossflag))
	print('** Smallest signal allowed: {}'.format(smallsig))
	print('** Signal percentile for normalisation: {}'.format(prctsig))
	if args.fte is not None:
		print('** Index of first echo in your multi-echo data: {}'.format(fte))
		print('** Number of echoes in your multi-echo data: {}'.format(nte))
	print('')
	print('')


	### Set random seeds
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	### Normalise data
	datatrain[datatrain<smallsig] = smallsig
	dataval[dataval<smallsig] = smallsig
	if args.ttar is not None:
		datattar[datattar<smallsig] = smallsig
		datavtar[datavtar<smallsig] = smallsig
	min_val = np.float32(smallsig)
	min_val_file = open(os.path.join(out_base_dir,'min_val.bin') ,'wb')
	pk.dump(min_val,min_val_file,pk.HIGHEST_PROTOCOL)      
	min_val_file.close()

	if args.ttar is not None:
		max_val = np.float32(np.percentile(datattar,prctsig))
	else:
		max_val = np.float32(np.percentile(datatrain,prctsig))
	max_val_file = open(os.path.join(out_base_dir,'max_val.bin') ,'wb')
	pk.dump(max_val,max_val_file,pk.HIGHEST_PROTOCOL)      
	max_val_file.close()

	datatrain = np.float32( (datatrain - min_val) / (max_val - min_val) )
	dataval = np.float32( (dataval - min_val) / (max_val - min_val) )

	if args.ttar is not None:
		datattar = np.float32( (datattar - min_val) / (max_val - min_val) )
		datavtar = np.float32( (datavtar - min_val) / (max_val - min_val) )	

	### Create mini-batches on training data with data loader
	loadertrain = DataLoader(datatrain, batch_size=mbatch, shuffle=True, num_workers=nwork)
	if args.ttar is not None:
		loadertrain = DataLoader(np.concatenate((datatrain,datattar),axis=1), batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	### Allocate memory for selected measurement indices
	meas_idx = np.zeros((noepoch,nobatch,dsel)) + np.nan
	meas_weight = np.zeros((noepoch,nobatch,nhidden_sel[0])) + np.nan

	### Instantiate the network and training objects, and save the intantiated network
	nnet = sardunet.sardunet_v1(nhidden_sel,psel,dsel,nhidden_pred,ppred).cpu()      # Instantiate neural network
	nnet.selector_update = True                                                  # Make sure selection indices get updated as training starts
	if args.fte is not None:
		nnet.selector_mechoFirstTe = fte                                             # Index of first echo
		nnet.selector_mechoNTe = nte                                                 # Number of echoes
	nnet.train()                                                                 # Set network to training mode (activates dropout)
	if(lossflag==2):
		nnetloss = nn.MSELoss()                                                      # Loss: L2
	elif(lossflag==1):
		nnetloss = nn.L1Loss()                                                       # Loss: L1
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'nnet_epoch0.pth') )    # Save network at epoch 0 (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'nnet_epoch0.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()


	### Run training
	# Loop over epochs
	loss_val_prev = np.inf
	for epoch in range(noepoch):
	    
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch))
		print('')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for signals in loadertrain:


			# Pass the mini-batch through the network and store the training loss
			if args.ttar is None: 
				output = nnet( Tensor(signals) )                                  # Pass data through full net   
				lossmeas_train = nnetloss(Tensor(output)*(max_val - min_val) + min_val, Tensor(signals)*(max_val - min_val) + min_val)  # Training loss (non-normalised loss)
			else:
				output = nnet( Tensor(signals[:,0:nmeas_train]) )                                  # Pass data through full net   
				lossmeas_train = nnetloss(Tensor(output)*(max_val - min_val) + min_val, Tensor(signals[:,nmeas_train:nmeas_train+nmeas_ttar])*(max_val - min_val) + min_val)  # Training loss (non-normalised loss)

			meas_idx[epoch,minibatch_id,:] = (nnet.selector_indices).detach().numpy()      # Selector indices
			meas_weight[epoch,minibatch_id,:] = (nnet.selector_weights).detach().numpy()   # Selector weights

			# Back propagation
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			# Store loss for the current mini-batch of training
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data)

			# Update mini-batch counter
			minibatch_id = minibatch_id + 1
		
		
		# Run validation
		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
		dataval_nnet = nnet( Tensor(dataval) )         # Output of full network
		if args.vtar is None:
			lossmeas_val = nnetloss( Tensor(dataval_nnet)*(max_val - min_val) + min_val , Tensor(dataval)*(max_val - min_val) + min_val )    # Validation loss (non-normalised loss)
		else:
			lossmeas_val = nnetloss( Tensor(dataval_nnet)*(max_val - min_val) + min_val , Tensor(datavtar)*(max_val - min_val) + min_val )    # Validation loss (non-normalised loss)
		# Store validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)

		# Save trained network at current epoch if validation loss has decreased
		nnet.selector_update = False
		if(Tensor.numpy(lossmeas_val.data)<=loss_val_prev):
			print('             ... validation loss has decreased. Saving net...')
			# Save network
			torch.save( nnet.state_dict(), os.path.join(out_base_dir,'nnet_lossvalmin.pth') )
			nnet_file = open(os.path.join(out_base_dir,'nnet_lossvalmin.bin'),'wb')
			pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
			nnet_file.close()
			# Save information on the epoch
			nnet_text = open(os.path.join(out_base_dir,'nnet_lossvalmin.info'),'w')
			nnet_text.write('Epoch {} (indices starting from 0)'.format(epoch));
			nnet_text.close();
			# Save indices of selected measurements as a text file
			savemeas = (nnet.selector_indices).detach().numpy()
			np.savetxt(os.path.join(out_base_dir,'nnet_lossvalmin_measidx.txt'), savemeas, fmt='%d', delimiter=',')
			# Update value of best validation loss so far
			loss_val_prev = Tensor.numpy(lossmeas_val.data)
			# Save validation signals
			dataval_nnet = dataval_nnet.detach().numpy()
			dataval_nnet = min_val + (max_val - min_val)*dataval_nnet
			dataval_nnet_file = open(os.path.join(out_base_dir,'nnet_lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()
		nnet.selector_update = True

		# Set network back to training mode
		nnet.train()

		# Print some information
		print('')
		print('             TRAINING INFO:')
		print('             Trainig loss: {:.12f}; validation loss: {:.12f}'.format(Tensor.numpy(lossmeas_train.data), Tensor.numpy(lossmeas_val.data)) )
		print('')

	# Save the final network
	nnet.selector_update = False
	nnet.eval()
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'nnet_epoch{}.pth'.format(noepoch)) )
	nnet_file = open(os.path.join(out_base_dir,'nnet_epoch{}.bin'.format(noepoch)),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	# Save the training and validation loss
	losstrain_file = open(os.path.join(out_base_dir,'losstrain.bin'),'wb')
	pk.dump(losstrain,losstrain_file,pk.HIGHEST_PROTOCOL)      
	losstrain_file.close()

	lossval_file = open(os.path.join(out_base_dir,'lossval.bin'),'wb')
	pk.dump(lossval,lossval_file,pk.HIGHEST_PROTOCOL)      
	lossval_file.close()

	# Save the selected measurement indices and weights
	meas_idx_file = open(os.path.join(out_base_dir,'meas_idx.bin'),'wb')
	pk.dump(meas_idx,meas_idx_file,pk.HIGHEST_PROTOCOL)      
	meas_idx_file.close()

	meas_weight_file = open(os.path.join(out_base_dir,'meas_weight.bin'),'wb')
	pk.dump(meas_weight,meas_weight_file,pk.HIGHEST_PROTOCOL)      
	meas_weight_file.close()

	# Save indices of selected measurements as a text file
	savemeas = (nnet.selector_indices).detach().numpy()
	np.savetxt(os.path.join(out_base_dir,'nnet_epoch{}_measidx.txt'.format(noepoch)), savemeas, fmt='%d', delimiter=',')
	

