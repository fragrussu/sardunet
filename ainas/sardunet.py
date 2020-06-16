import sys
import torch
import numpy as np
from torch import nn
from torch import Tensor


class sardunet_v1(nn.Module):
	''' Select and retrieve for direct upsaling (SARDU)-Net: find measures that optimally support prediction

	    A SARDU-Net is made of two neural networks: a selector, choosing with measurements to keep given 
            a rich MRI protocol; a selector, that tries to retrieve the richly sampled signal from the subset

	    Author: Francesco Grussu, University College London
			               Queen Square Institute of Neurology
					Centre for Medical Image Computing
		                       <f.grussu@ucl.ac.uk><francegrussu@gmail.com>

	    # Code released under BSD Two-Clause license
	    #
	    # Copyright (c) 2020 University College London. 
	    # All rights reserved.
	    #
	    # Redistribution and use in source and binary forms, with or without modification, are permitted 
	    # provided that the following conditions are met:
	    # 
	    # 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
	    # disclaimer.
	    # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
	    # disclaimer in the documentation and/or other materials provided with the distribution.
	    # 
	    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
	    # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
	    # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
	    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
	    # USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
	    # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
	    # OF THE POSSIBILITY OF SUCH DAMAGE.
	    # 
	    # The views and conclusions contained in the software and documentation are those
	    # of the authors and should not be interpreted as representing official policies,
	    # either expressed or implied, of the FreeBSD Project.

	'''    
	def __init__(self,n_selector,pdrop_selector,ds_factor,n_predictor,pdrop_predictor):
		'''     Initialise a sardunet version 1 as:
			 
			 mynet = sardunet_v1(n_selector,pdrop_selector,ds_factor,n_predictor,pdrop_predictor)
			 * n_selector: numpy array storing the number of input neurons to each selector layer
			 * pdrop_selector: dropout probability of selector
			 * ds_factor: number of measurements in the downsampled protocol
			 * n_predictor: numpy array storing the number of input neurons to each predictor layer
			 * pdrop_predictor: dropout probability of predictor
			 * mynet: new SARDU-net
			
			 Once a SARDU-net has been created, you can flag whether there are multi-echo measurements
			 to be analysed in block. In that case, SARDU-net will only find the maximum echo time TE
			 and will keep all the TEs. Do so by setting the fields selector_multiecho.first_te and
			 selector_multiecho.no_te of your net. For example, if first echo is measured 12th and 
			 there are 32 echoes:
			 
			 * mynet.selector_mechoFirstTe = 12
			 * mynet.selector_mechoNTe = 32

			 Information about measurement selection is stored in the following fields of a SARDU-Net:
			 * mynet.selector_indices: stores the indices of the selected measurements
			 * mynet.selector_weights: stores measurement-wise weights for the selected measurements
			 * mynet.selector_downsampsize: stores value of input parameter ds_factor
			 * mynet.selector_layers: stores the layers of the selector net
			 * mynet.predictor_layers: stores the layers of the predictor net
			 * mynet.selector_nneurons: stores input parameter n_selector 
			 * mynet.predictor_nneurons: stores input parameter n_predictor 
			 * mynet.selector_update: true when indices of measurements to select are being updated 
						   (i.e. during training), false otherwise	
			
		'''
		super(sardunet_v1, self).__init__()

		## Generate hidden layers of selector
		if (n_selector[-1]!=n_selector[0]):
			raise RuntimeError('the number of output neurons of the selector (you set {}) must be the same as the input (you set {})'.format(n_selector[-1],n_selector[0]))
		if (ds_factor>=n_selector[-1]):		
			raise RuntimeError('the number of selected measurements (you set {}) must be the smaller than the number of input measurements (you set {})'.format(ds_factor,n_selector[0]))
		layerlist_sel = []
		nlayers_sel = n_selector.shape[0] - 1
		for ll in range(nlayers_sel):
			layerlist_sel.append(nn.Linear(n_selector[ll], n_selector[ll+1]))
			if(ll<(nlayers_sel-1)):
				layerlist_sel.append(nn.ReLU(True))                   # Do not add ReLU in last layer
				layerlist_sel.append(nn.Dropout(p=pdrop_selector))    # Do not add dropout in last layer
		self.selector_layers = nn.ModuleList(layerlist_sel)
		self.selector_nneurons = n_selector    # Number of hidden neurons per layer in the selector
		self.selector_downsampsize = ds_factor # Number of downsampled measurements
		self.selector_mechoFirstTe = np.nan    # Index of first TE (starting from 0) in multi-echo measurements to analyse in block (SARDU-net will only choose the maximum echo to keep) 
		self.selector_mechoNTe = np.nan        # Number of TEs in multi-echo measurements to analyse in block (SARDU-net will only choose the maximum echo to keep) 
		self.selector_indices = []
		self.selector_weights = []
		self.selector_update = True            # Flag to indicate whether we need to update the indices of the subset to select or not
		

		## Generate hidden layers of the predictor
		if(n_predictor[0]!=n_selector[-1]):		
			raise RuntimeError('the number of input neurons of the predictor (you set {}) must be the same as the output of the selector (you set {})'.format(n_predictor[0],n_selector[-1]))
		layerlist_pred = []
		nlayers_pred = n_predictor.shape[0] - 1
		for ll in range(nlayers_pred):
			layerlist_pred.append(nn.Linear(n_predictor[ll], n_predictor[ll+1]))
			if(ll<(nlayers_pred-1)):
				layerlist_pred.append(nn.ReLU(True))                      # Do not add ReLU in last layer
				layerlist_pred.append(nn.Dropout(p=pdrop_predictor))      # Do not add dropout in last layer
		self.predictor_layers = nn.ModuleList(layerlist_pred)
		self.predictor_nneurons = n_predictor # Number of hidden neurons per layer in the predictor


	def selector(self,x):
		''' Selector of a SARDU-Net

		    xout = mynet.selector(xin) 

		    * mynet: initialised SARDU-Net
		    * xin:   pytorch Tensor storing MRI signals from one or from multiple voxels (mini-bacth;   
			     for a mini-batch, xin has size voxels x measurements) 
		    * xout:  pytorch Tensor, same size as xin, storing MRI signals where non-selected 
			     measurements are zeroed while selected measurements are scaled by a 
			     measurement-dependent weight (if mynet.selector_update is true such weights are
			     calculated, while if mynet.selector_update is false then the weigths stored in
			     mynet.selector_weights are used)           	

		'''
		## Calculate new selection indices if selector_update is True
		if self.selector_update==True:

			xin = torch.clone(x)

			# Pass net through fully connected layers of the selector
			for mylayer in self.selector_layers:
				x = mylayer(x)

			# Obtain indices of most informative measurements and zero the others: single input case
			if x.dim()==1:
                
				# Softmin normalisation and thresholding
				layer_softmin = nn.Softmin(dim=0)
				x = layer_softmin(x)
				w_sort,w_idx = torch.sort(x,descending=True)
				x[ Tensor.numpy( w_idx[(self.selector_downsampsize):self.selector_nneurons[-1]] ) ] = 0.0

				# Account for multi-echo data to be analysed as one block if required				
				if(np.isnan(self.selector_mechoFirstTe)==False):
					nonzero_tes = int( np.sum( Tensor.numpy( x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+self.selector_mechoNTe]!=0 ) ) ) # Number of non-zero TEs
					# Keep only consecutive TEs up to the current number of non-zero TEs and use the same weight for each of them
					if(nonzero_tes!=0):
						x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+nonzero_tes] = torch.mean(x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+self.selector_mechoNTe])
						x[self.selector_mechoFirstTe+nonzero_tes:self.selector_mechoFirstTe+self.selector_mechoNTe] = 0.0


				# Get final set of selector weights
				x = x/torch.sum(x)
                
			# Obtain indices of most informative measurements: mini-batch case
			elif x.dim()==2:

				# Softmin normalisation and thresholding                
				layer_softmin = nn.Softmin(dim=1)
				x = layer_softmin(x)
				x = torch.mean(x,dim=0)
				w_sort,w_idx = torch.sort(x,descending=True)
				x[ Tensor.numpy( w_idx[(self.selector_downsampsize):(self.selector_nneurons[-1])] ) ] = 0.0

				# Account for multi-echo data to be analysed as one block if required				
				if(np.isnan(self.selector_mechoFirstTe)==False):
					nonzero_tes = int( np.sum( Tensor.numpy( x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+self.selector_mechoNTe]!=0 ) ) ) # Number of non-zero TEs
					# Keep only consecutive TEs up to the current number of non-zero TEs and use the same weight for each of them
					if(nonzero_tes!=0):
						x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+nonzero_tes] = torch.mean(x[self.selector_mechoFirstTe:self.selector_mechoFirstTe+self.selector_mechoNTe])
						x[self.selector_mechoFirstTe+nonzero_tes:self.selector_mechoFirstTe+self.selector_mechoNTe] = 0.0

				# Get final set of selector weights
				x = x/torch.sum(x)
                                            
			else:
				raise RuntimeError('input tensors need to be 1D or 2D; your data is {}D instead'.format(x.dim()))

			# Extract measurements with the newly calculated selector indices
			if xin.dim()==1:
				xout = xin * x
			elif xin.dim()==2:
				xout = xin * (x.repeat(xin.shape[0],1))
			else:
				raise RuntimeError('input tensors need to be 1D or 2D; your data is {}D instead'.format(xin.dim()))
              
                
			# Store updated selector indices and weights, accounting for multi-echo blocks if required 
			if(np.isnan(self.selector_mechoFirstTe)==False):
				w_idx_me = Tensor(np.arange(self.selector_nneurons[0],dtype=int))
				w_idx_me = w_idx_me.type(torch.LongTensor)
				self.selector_indices = w_idx_me[x!=0.0]
			else:
				w_sort_new,w_idx_new = torch.sort(w_idx[0:self.selector_downsampsize])
				self.selector_indices = w_sort_new
			self.selector_weights = x

		## Use old selection indices if selector_update is False
		elif self.selector_update==False:

			# Extract measurements
			wx = self.selector_weights
			if x.dim()==1:
				xout = x * wx
			elif x.dim()==2:
				xout = x * (wx.repeat(x.shape[0],1))
			else:
				raise RuntimeError('input tensors need to be be 1D or 2D; your data is {}D instead'.format(x.dim()))

		## Error if selector_update is set to bad values
		else:
			raise RuntimeError('field selector_update is {} but must be either True or False'.format(self.selector_update))
		
		# Return output made of the selection of only some input measurements, weighted by some weights
		return xout


    
	def predictor(self, x):   
		''' Predictor of a SARDU-Net

		    xpred = mynet.predictor(xsel) 

		    * mynet:  SARDU-Net
		    * xsel:   pytorch Tensor storing the output of the selector net 
		    * xpred:  pytorch Tensor, same size as xin, storing the estimating of the richly-
			      sampled MR signals obtianed from the sets of measurements selected by the
			      selector
		'''        
		# Pass the salient measurements through a standard multi-layer fully-connected net
		for mylayer in self.predictor_layers:
			x = mylayer(x)
		return x
    

    
	def forward(self, x):
		''' Forward pass of a SARDU-Net

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  SARDU-Net
		    * x:      pytorch Tensor storing richly-sampled MRI signals from one or from multiple voxels 
			      (mini-bacth; for a mini-batch, xin has size voxels x measurements)
		    * y:      pytorch Tensor storing an estimate of x obtained from a subset of measurements, or 
			      of a set of measurements even more richly-sampled than x
		'''  

		# Select relevant measurements
		x = self.selector(x)

		# Predict output
		x = self.predictor(x)

		# Return prediction           
		return x


