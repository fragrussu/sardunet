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
	parser = argparse.ArgumentParser(description='This program uses a trained SARDU-net to downsample a multi-contrast MRI scan on a voxel-by-voxel basis, keeping only salient MRI measurements. It requires as input a trained SARDU-net, a 4D NIFTI file with an MRI scan to downsample and outputs the downsampled MRI scan as a 4D NIFTI. Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')	
	parser.add_argument('nifti_in', help='path to a 4D NIFTI file storing a multi-contrast MRI scan to downsample')
	parser.add_argument('nifti_out', help='path of the output 4D NIFTI file that will store the downsampled MRI scan')
	parser.add_argument('sardunet_file', help='path to a pickle binary file containing a trained SARDU-net')
	parser.add_argument('--bits', metavar='<value>',  default='32', help='number of bits per voxel in the output file (default 32 bits; choose either 32 or 64 bits)')
	args = parser.parse_args()

	### Print some information
	print('')
	print('********************************************************************')
	print('                DOWNSAMPLE WITH A S.A.R.D.U.-NET                    ')
	print('********************************************************************')
	print('')
	print('** Called on 4D Nifti file: {}'.format(args.nifti_in))
	print('** SARDU-net file: {}'.format(args.sardunet_file))
	print('** 4D output Nifti file: {}'.format(args.nifti_out))
	print('')	

	### Load input NIFTI
	print('')
	print('      ... loading data ...')
	sin_obj = nib.load(args.nifti_in)
	sin_header = sin_obj.header
	sin_affine = sin_header.get_best_affine()
	sin_data = sin_obj.get_data()
	sin_dims = sin_data.shape
	imgsize = sin_data.shape
	imgsize = np.array(imgsize)
	sin_data = np.array(sin_data,'float64')

	if imgsize.size!=4:
		print('')
		print('ERROR: the input 4D NIFTI file {} is not actually not 4D. Exiting with 1...'.format(args.nifti_in))				 
		print('')
		sys.exit(1)


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

	### Get indices of measurements to keep
	idx = Tensor.numpy(net.selector_indices)  # Indices of measurements to keep
	idx_sort = np.sort(idx)              # Make sure indices are sorted

	### Downsample scan
	print('')
	print('      ... downsampling scan.')
	print('')
	print('          The following measurements out of {} will be kept (indices start from 0):'.format(sin_dims[3]))
	print('          {}'.format(idx_sort))
	sout_data = sin_data[:,:,:,idx_sort]

	### Save downsampled scan as NIFTI
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

