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
import nibabel as nib
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
	parser = argparse.ArgumentParser(description='This program uses a trained SARDU-net to upsample a multi-contrast MRI scan on a voxel-by-voxel basis. It requires as input a SARDU-net, a 4D NIFTI file with an MRI scan to upsample, a mask (optional) and outputs the upsampled MRI scan as a 4D NIFTI. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')	
	parser.add_argument('nifti_in', help='path to a 4D NIFTI file storing a multi-contrast MRI scan to upsample')
	parser.add_argument('nifti_out', help='path of the output 4D NIFTI file that will store the upsampled MRI scan')
	parser.add_argument('sardunet_file', help='path to a pickle binary file containing a trained SARDU-net')
	parser.add_argument('sigmax_file', help='path to a pickle binary file containing the value of the maximum signal smax used for normalisation (smax = 1 in normalised signal space)')
	parser.add_argument('sigmin_file', help='path to a pickle binary file containing the value of the minimum signal smin used for normalisation (smin = 0 in normalised signal space)')
	parser.add_argument('--mask', metavar='<nifti>', help='path to a mask flagging with 1 voxels to analysise and with 0 voxels to ignore (the output NIFTI will contain 0 in voxels to ignore)')
	parser.add_argument('--bits', metavar='<value>',  default='32', help='number of bits per voxel in the output file (default 32 bits; choose either 32 or 64 bits)')
	args = parser.parse_args()

	### Print some information
	print('')
	print('********************************************************************')
	print('                   UPSAMPLE WITH S.A.R.D.U.-NET                     ')
	print('********************************************************************')
	print('')
	print('** Called on 4D Nifti file: {}'.format(args.nifti_in))
	print('** SARDU-net file: {}'.format(args.sardunet_file))
	print('** 4D output Nifti file: {}'.format(args.nifti_out))
	print('** File storing  maximum signal for linear scaling: {}'.format(args.sigmax_file))
	print('** File storing  minimum signal for linear scaling: {}'.format(args.sigmin_file))
	print('')	

	### Load input NIFTI
	print('')
	print('      ... loading data ...')
	sin_obj = nib.load(args.nifti_in)
	sin_header = sin_obj.header
	sin_affine = sin_header.get_best_affine()
	sin_data = sin_obj.get_fdata()
	sin_dims = sin_data.shape
	imgsize = sin_data.shape
	imgsize = np.array(imgsize)
	sin_data = np.array(sin_data,'float64')

	if imgsize.size!=4:
		print('')
		print('ERROR: the input 4D NIFTI file {} is not actually not 4D. Exiting with 1...'.format(args.nifti_in))				 
		print('')
		sys.exit(1)


	### Deal with optional arguments: mask
	if isinstance(args.mask, str)==1:
		# A mask for SARDU-net has been provided
		mask_obj = nib.load(args.mask)
		mask_dims = mask_obj.get_shape()		
		mask_header = mask_obj.header
		mask_affine = mask_header.get_best_affine()			
		# Make sure the mask is a 3D file
		mask_data = mask_obj.get_fdata()
		masksize = mask_data.shape
		masksize = np.array(masksize)
		
		if masksize.size!=3:
			print('')
			print('WARNING: the mask file {} is not a 3D Nifti file. Ignoring mask...'.format(mask_nifti))				 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		elif ( (np.sum(sin_affine==mask_affine)!=16) or (sin_dims[0]!=mask_dims[0]) or (sin_dims[1]!=mask_dims[1]) or (sin_dims[2]!=mask_dims[2]) ):
			print('')
			print('WARNING: the geometry of the mask file {} does not match that of the input data. Ignoring mask...'.format(args.mask))					 
			print('')
			mask_data = np.ones(imgsize[0:3],'float64')
		else:
			mask_data = np.array(mask_data,'float64')
			# Make sure mask data is a numpy array
			mask_data[mask_data>0] = 1
			mask_data[mask_data<=0] = 0

	else:
		# A mask for fitting has not been provided
		mask_data = np.ones(imgsize[0:3],'float64')

	### Bits per pixel
	nbit = int(args.bits)
	if( (nbit!=32) and (nbit!=64) ):
		print('')
		print('ERROR: the voxel bit depth must be either 32 or 64. You set {}. Exiting with 1...'.format(args.bits))				 
		print('')
		sys.exit(1)

	#### Load SARDU-net in evaluation mode
	h = open(args.sardunet_file,'rb') 
	net = pk.load(h)
	h.close()
	net.eval()

	#### Load normalisation factors
	h = open(args.sigmax_file,'rb') 
	smax = pk.load(h)
	h.close()

	h = open(args.sigmin_file,'rb') 
	smin = pk.load(h)
	h.close()

	### Allocate memory for network inputs and outputs
	if(imgsize[3]!=net.selector_downsampsize):
		print('')
		print('ERROR: the input 4D NIFTI file {} has a number of measurements that does not match that of SARDU-net. Exiting with 1...'.format(args.nifti_in))				 
		print('')
		sys.exit(1)
	idx = Tensor.numpy(net.selector_indices)  # Indices of measurements
	idx_sort = np.sort(idx)    # Indices to guide zero-filling 
	nin = net.predictor_nneurons[0]    # Number of neurons in first layer
	nout = net.predictor_nneurons[-1]    # Number of output measurements
	sout_data = np.zeros((imgsize[0],imgsize[1],imgsize[2],nout))

	### Loop to predict signals
	print('')
	print('     ... predicting MRI signals with SARDU-net ...')
	for xx in range(0, imgsize[0]):
		for yy in range(0, imgsize[1]):
			for zz in range(0, imgsize[2]):

				# Get voxel within mask
				if(mask_data[xx,yy,zz]==1):

					# Get qMRI signal
					myvoxel = sin_data[xx,yy,zz,:]
					# Normalise qMRI signal
					myvoxel[myvoxel<smin] = smin
					myvoxel = np.float32( (myvoxel - smin) / (smax - smin) )
					# Zero-fill missing measurements
					myvoxel_zeroed = np.zeros(nin)
					myvoxel_zeroed = np.float32(myvoxel_zeroed)
					for qq in range(0,imgsize[3]):
						myvoxel_zeroed[idx_sort[qq]] = myvoxel[qq]
					# Pass voxel through SARDU-net
					myvoxel_up = net(Tensor(myvoxel_zeroed))
					myvoxel_up = myvoxel_up.detach().numpy()
					# Bring voxel back to original signal space
					myvoxel_up = myvoxel_up*(smax - smin) + smin
					# Store voxel
					sout_data[xx,yy,zz,:] = myvoxel_up
		
	
	# Save the predicted NIFTI file as output
	print('')
	print('     ... saving output file as 4D NIFTI ...')
	buffer_header = sin_obj.header
	if(nbit==64):
		buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
	elif(nbit==32):
		buffer_header.set_data_dtype('float32')   # Make sure we save output files float64, even if input is not
	sout_obj = nib.Nifti1Image(sout_data,sin_obj.affine,buffer_header)
	nib.save(sout_obj, args.nifti_out)

	print('')
	print('     Done!')
	print('')

	sys.exit(0)

