#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created by  : Niclas Rieger
# Created on  : Tue Jun 30 10:27:30 2020
# =============================================================================
""" 
Maximum Covariance Analysis of two data fields 
with complexified data fields and rotation
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
from scipy.signal import hilbert
import matplotlib.pylab as plt
import matplotlib as mplt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from rotation import promax


# =============================================================================
#%% MCA
# =============================================================================
class MCA(object):
	"""Perform maximum covariance analysis (MCA) for two `np.ndarray` data fields.
	
	MCA is principal component analysis (PCA) generalized 
	for two input fields (left, right). If both data fields are the same, 
	it is equivalent to PCA.

	Parameters
	----------
	left : ndarray
		Left input data. First dimension needs to be time. 
	right : ndarray, optional
		Right input data. First dimension needs to be time.
		If none is provided, automatically, right field is assumed to be
		the same as left field. In this case, MCA reducdes to normal PCA. 
		The default is None.
	normalize : boolean, optional
		Input data is normalized to unit variance. The default is True.
	
	
	Examples
	--------
	Let `data1` and `data2` be some geophysical fields (e.g. SST and pressure). 
	To perform PCA use: 
	
	>>> pca = MCA(data1)
	>>> pca.solve()
	>>> pcs,_ = pca.pcs()
	
	To perform MCA use:
	
	>>> mca = MCA(data1, data2)
	>>> mca.solve()
	>>> pcsData1, pcsData2 = mca.pcs()		
	"""	
	
	def __init__(self, left, right=None, normalize=True):
		"""Load data fields and store information about data size/shape.
		
		Parameters
		----------
		left : ndarray
			Left input data. First dimension needs to be time. 
		right : ndarray, optional
			Right input data. First dimension needs to be time.
			If none is provided, automatically, right field is assumed to be
			the same as left field. In this case, MCA reducdes to normal PCA. 
			The default is None.
		normalize : boolean, optional
			Input data is normalized to unit variance. The default is True.
		

		"""
		self.__left  = left.copy()
		self.__right = self.__getRightField(left,right)
			
		assert(self.__isArray(self.__left))
		assert(self.__isArray(self.__right))
		assert(self.__hasSameTimeDimensions(self.__left, self.__right))
		

		self.__observations 		= self.__left.shape[0]
		self.__originalShapeLeft 	= self.__left.shape[1:]
		self.__originalShapeRight 	= self.__right.shape[1:]
		
		self.__variablesLeft 		= np.product(self.__originalShapeLeft)
		self.__variablesRight 		= np.product(self.__originalShapeRight)
		 
		# create 2D matrix in order to perform PCA
		self.__left 	= self.__left.reshape(self.__observations,self.__variablesLeft)
		self.__right 	= self.__right.reshape(self.__observations,self.__variablesRight)
		
		# center input data to zero mean (remove mean)
		self.__left 	= self.__centerData(self.__left)
		self.__right 	= self.__centerData(self.__right)
		
		# normalize input data to unit variance
		if (normalize):
			self.__left  = self.__normalizeData(self.__left)
			self.__right = self.__normalizeData(self.__right)

		# remove NaNs in data fields
		self.__noNanIndexLeft 	= np.where(~(np.isnan(self.__left[0])))[0]
		self.__noNanIndexRight 	= np.where(~(np.isnan(self.__right[0])))[0]

		self.__noNanDataLeft 	= self.__left[:,self.__noNanIndexLeft]
		self.__noNanDataRight 	= self.__right[:,self.__noNanIndexRight]
		
		assert(self.__isNotEmpty(self.__noNanDataLeft))
		assert(self.__isNotEmpty(self.__noNanDataRight))
		
		self.__rotatedSolution = False


	def __getRightField(self,left,right):
		"""Copy left field if no right field is provided.
		
		Basically, this defines whether MCA or PCA is performed.
		"""
		if right is None:
			return left.copy()	
		else:
			return right.copy() 		
		
		
	def __isArray(self,data):
		if (isinstance(data,np.ndarray)):
			return True
		else:
			raise TypeError('Data needs to be np.ndarray.')
	
	
	def __hasSameTimeDimensions(self, left, right):
		if (left.shape[0] == right.shape[0]):
			return True
		else:
			raise ValueError('Both input fields need to have same time dimensions.')
	
	
	def __centerData(self, data, normalize=False):
		"""Remove the mean of an array along the first dimension."""
		return data - data.mean(axis=0)
		
	
	def __normalizeData(self,data):
		"""Normalize the array along the first dimension (divide by std)."""
		return data / data.std(axis=0)


	def __isNotEmpty(self, index):
		if (index.size > 0):
			return True
		else:
			raise ValueError('Input field is empty or contains NaN only.')
	

	def solve(self, useHilbert=False):
		"""Solve eigenvalue equation by performing SVD on covariance matrix.
		
		Parameters
		----------
		useHilbert : boolean, optional
			Use Hilbert transform to complexify the input data fields
			in order to perform complex PCA/MCA
		"""
		print('Start analysis...',flush=True)
		
		# complexify input data via Hilbert transfrom
		if (useHilbert):
			print('Apply Hilbert transform...', flush=True)
			self.__noNanDataLeft 	= hilbert(self.__noNanDataLeft,axis=0)
			self.__noNanDataRight 	= hilbert(self.__noNanDataRight,axis=0)
			
		print('Build Covariance matrix...',flush=True)
		kernel = self.__noNanDataLeft.conjugate().T @ self.__noNanDataRight / self.__observations
			
		# solve eigenvalue problem
		print('Perform SVD...',flush=True)
		VLeft, eigenvalues, VTRight = np.linalg.svd(kernel, full_matrices=False) 
		VRight = VTRight.conjugate().T
		
		S = np.sqrt(np.diag(eigenvalues) * self.__observations)
		Si = np.diag(1./np.diag(S)) 
		
		self.__eigenvalues = eigenvalues
		self.__eigensum = eigenvalues.sum()
		
		# standardized EOF fields
		self.__VLeft 	= VLeft
		self.__VRight 	= VRight
		
		# loadings // EOF fields
		self.__LLeft 	= self.__VLeft @ S 
		self.__LRight 	= self.__VRight @ S
		
		# get PC scores by projecting data fields on loadings 
		self.__ULeft 	= self.__noNanDataLeft @ self.__VLeft @ Si
		self.__URight 	= self.__noNanDataRight @ self.__VRight @ Si
				
		print('Finished!', flush=True)


	def rotate(self, nRotations, power=1, tol=1e-5):
		"""Perform Promax rotation on the first `n` EOFs.
		
		Promax rotation (Hendrickson & White 1964) is an oblique rotation which 
		seeks to find `simple structures` in the EOFs. It transforms the EOFs 
		via an orthogonal Varimax rotation (Kaiser 1958) followed by the Promax 
		equation. If `power=1`, Promax reduces to Varimax rotation. In general,
		a Promax transformation breaks the orthogonality of EOFs and introduces 
		some correlation between PCs. 

		Parameters
		----------
		nRotations : int
			Number of EOFs to rotate.
		power : int, optional
			Power of Promax rotation. The default is 1.
		tol : float, optional
			Tolerance of rotation process. The default is 1e-5.

		Raises
		------
		ValueError
			If number of rotations are <2.

		Returns
		-------
		None.

		"""
		print('Rotate EOFs...', flush=True)
		if(nRotations < 2):
			raise ValueError('nRotations must be >=2')
		if(power<1):
			raise ValueError('Power must be >=1')

		# rotate loadings (Cheng and Dunkerton 1995)
		L = np.concatenate((self.__LLeft[:,:nRotations], self.__LRight[:,:nRotations]))
		Lr, R, Phi = promax(L, power, maxIter=1000, tol=tol)
		LLeft 	= Lr[:self.__noNanIndexLeft.size,:]
		LRight 	= Lr[self.__noNanIndexLeft.size:,:]
		
		# calculate variance/reconstruct "eigenvalues"
		wLeft = np.linalg.norm(LLeft,axis=0) 
		wRight = np.linalg.norm(LRight,axis=0)
		variance = wLeft * wRight / self.__observations
		varIdx = np.argsort(variance)[::-1]
		
		# pull loadings from EOFs
		VLeft 	= LLeft / wLeft
		VRight 	= LRight / wRight
		
		# rotate PC scores   
		# If rotation is orthogonal: R.T = R
		# If rotation is oblique (p>1): R^(-1).T = R
		if(power==1):
			ULeft 	= self.__ULeft[:,:nRotations] @ R
			URight 	= self.__URight[:,:nRotations] @ R
		else:
			ULeft 	= self.__ULeft[:,:nRotations] @ np.linalg.pinv(R).conjugate().T
			URight 	= self.__URight[:,:nRotations] @ np.linalg.pinv(R).conjugate().T		
		
		
		# store rotated pcs, eofs and "eigenvalues"
		# and sort according to described variance
		self.__eigenvalues 	= variance[varIdx]
		# Standardized EOFs
		self.__VLeft 		= VLeft[:,varIdx]
		self.__VRight 		= VRight[:,varIdx]
		# EOF loadings
		self.__LLeft 		= LLeft[:,varIdx]
		self.__LRight 		= LRight[:,varIdx]
		# Standardized PC scores
		self.__ULeft 		= ULeft[:,varIdx]
		self.__URight 		= URight[:,varIdx]
		
		# store rotation matrix and correlation matrix of PCs
		self.__rotationMatrix 		= R
		self.__correlationMatrix 	= Phi[varIdx,varIdx]
		self.__rotatedSolution 		= True


	def correlationMatrix(self):
		"""
		Return the correlation matrix of rotated PCs.

		Returns
		-------
		ndarray
			Correlation matrix.

		"""
		if (self.__rotatedSolution):
			return self.__correlationMatrix
		else:
			raise RuntimeError('Correlation matrix does not exist since PCs were not rotated.')


	def eigenvalues(self,n=None):
		"""Return the first `n` eigenvalues.

		Parameters
		----------
		n : int, optional
			Number of eigenvalues to return. The default is 5.

		Returns
		-------
		values : ndarray
			Eigenvalues of PCA.
		error : ndarray
			Uncertainty of eigenvalues according to North's rule of thumb.

		"""
		values = self.__eigenvalues[:n]
		# error according to North's Rule of Thumb
		error = np.sqrt(2/self.__observations) * values
		
		return values, error


	def explainedVariance(self, n=None):
		"""Return the described variance of the first `n` PCs.

		Parameters
		----------
		n : int, optioal
			Number of PCs to return. The default is None.

		Returns
		-------
		desVar : ndarray
			Described variance of each PC.
		desVarErr : ndarray
			Associated uncertainty according to North's `rule of thumb`.

		"""
		values, error = self.eigenvalues(n)
		desVar 		= values / self.__eigensum * 100
		desVarErr 	= error / self.__eigensum * 100
		return desVar, desVarErr
	
	
	def pcs(self, n=None):
		"""Return the first `n` PCs.

		Parameters
		----------
		n : int, optional
			Number of PCs to be returned. The default is None.

		Returns
		-------
		pcsLeft : ndarray
			PCs associated with left input field.
		pcsRight : ndarray
			PCs associated with right input field.

		"""
		pcsLeft 	= self.__ULeft[:,:n]
		pcsRight 	= self.__URight[:,:n]
		return pcsLeft, pcsRight
	
	
	def eofs(self, n=None):
		"""Return the first `n` EOFs.

		Parameters
		----------
		n : int, optional
			Number of EOFs to be returned. The default is None.

		Returns
		-------
		eofsLeft : ndarray
			EOFs associated with left input field.
		eofsRight : ndarray
			EOFs associated with right input field.

		"""
		if n is None:
			n = self.__eigenvalues.size
		
		# create data fields with original NaNs
		dtype = self.__VLeft.dtype
		eofsLeft  	= np.zeros([self.__variablesLeft, n],dtype=dtype) * np.nan
		eofsRight  	= np.zeros([self.__variablesRight, n],dtype=dtype) * np.nan
		
		eofsLeft[self.__noNanIndexLeft,:] = self.__VLeft[:,:n]
		eofsRight[self.__noNanIndexRight,:] = self.__VRight[:,:n]
		
		# reshape data fields to have original input shape
		eofsLeft 	= eofsLeft.reshape(self.__originalShapeLeft + (n,))
		eofsRight 	= eofsRight.reshape(self.__originalShapeRight + (n,))
		return eofsLeft, eofsRight




# =============================================================================
#%% xMCA
# =============================================================================

class xMCA(MCA):
	"""Perform maximum covariance analysis (MCA) for two `xr.DataArray` data fields.
	
	MCA is principal component analysis (PCA) generalized 
	for two input fields (left, right). If both data fields are the same, 
	it is equivalent to PCA.

	Parameters
	----------
	left : ndarray
		Left input data. First dimension needs to be time. 
	right : ndarray, optional
		Right input data. First dimension needs to be time.
		If none is provided, automatically, right field is assumed to be
		the same as left field. In this case, MCA reducdes to normal PCA. 
		The default is None.
	normalize : boolean, optional
		Input data is normalized to unit variance. The default is True.
	
	
	Examples
	--------
	Let `data1` and `data2` be some geophysical fields (e.g. SST and pressure). 
	To perform PCA use: 
	
	>>> pca = MCA(data1)
	>>> pca.solve()
	>>> pcs,_ = pca.pcs()
	
	To perform MCA use:
	
	>>> mca = MCA(data1, data2)
	>>> mca.solve()
	>>> pcsData1, pcsData2 = mca.pcs()		
	"""		
	
	def __init__(self, left, right=None, normalize=True):
		"""Load data fields and store information about data size/shape.
		
		Parameters
		----------
		left : ndarray
			Left input data. First dimension needs to be time. 
		right : ndarray, optional
			Right input data. First dimension needs to be time.
			If none is provided, automatically, right field is assumed to be
			the same as left field. In this case, MCA reducdes to normal PCA. 
			The default is None.
		normalize : boolean, optional
			Input data is normalized to unit variance. The default is True.

		Returns
		-------
		None.

		"""
		self.__left  = left.copy()
		
		# for univariate case (normal PCA) left = right
		if right is None:
			self.__right = self.__left	
		else:
			self.__right = right.copy()
		
		assert(self.__isXarray(self.__left))
		assert(self.__isXarray(self.__right))

		# store meta information of time steps
		self.__timeSteps 	= self.__left.coords['time'].values
		# store meta information of coordinates
		self.__lonsLeft 	= self.__left.coords['lon'].values
		self.__lonsRight 	= self.__right.coords['lon'].values
		self.__latsLeft 	= self.__left.coords['lat'].values
		self.__latsRight 	= self.__right.coords['lat'].values
								
		dataLeft 	= self.__left.data
		dataRight 	= self.__right.data		
		
		# constructor of base class for np.ndarray
		MCA.__init__(self, dataLeft, dataRight, normalize)
		
		
	def __isXarray(self, data):
		"""Check if data is of type `xr.DataArray`.

		Parameters
		----------
		A : DataArray
			Input data.

		Returns
		-------
		bool
			Input data is of type `DataArray`.
			
		"""
		if (isinstance(data,xr.DataArray)):
			return True
		else:
			raise TypeError('Input data must be xarray.DataArray.')


	def eigenvalues(self, n=None):
		"""Return first `n` eigenvalues of the PCA.

		Parameters
		----------
		n : int, optional
			Number of eigenvalues to return. If none, then all eigenvalues are returned. 
			The default is None.

		Returns
		-------
		DataArray
			Eigenvalues of PCA.
		DataArray
			Uncertainty of eigenvalues according to North's rule of thumb.

		"""
		# for n=Nonr, all eigenvalues are returned			
		val, err = MCA.eigenvalues(self, n)
		
		# if n is not provided, take all eigenvalues
		if n is None:
			n = val.size
			
		modes = list(range(1,n+1))
		values = xr.DataArray(val,
					dims 	= ['mode'],
					coords 	= {'mode' : modes},
					name 	= 'eigenvalues')
		error = xr.DataArray(err,
					dims 	= ['mode'],
					coords 	= {'mode' : modes},
					name 	= 'uncertainty of eigenvalues')

		return values, error
	
	
	def explainedVariance(self, n=None):
		"""Return the described variance of the first `n` PCs.

		Parameters
		----------
		n : int, optioal
			Number of PCs to return. The default is None.

		Returns
		-------
		DataArray
			Described variance of each PC.
		DataArray
			Associated uncertainty according to North's `rule of thumb`.

		"""
		desVar, desVarErr 	= MCA.explainedVariance(self, n)
	
		# if n is not provided, take all eigenvalues
		if n is None:
			n = desVar.size
			
		modes = list(range(1,n+1))
		values = xr.DataArray(desVar,
					dims 	= ['mode'],
					coords 	= {'mode' : modes},
					name 	= 'described variance')
		error = xr.DataArray(desVarErr,
					dims 	= ['mode'],
					coords 	= {'mode' : modes},
					name 	= 'uncertainty of described variance')

		return desVar, desVarErr

	
	def pcs(self, n=None):
		"""Return first `n` PCs.

		Parameters
		----------
		n : int, optional
			Number of PCs to return. If none, then all PCs are returned. 
			The default is None.

		Returns
		-------
		DataArray
			PCs of left input field.
		DataArray
			PCs of right input field.

		"""
		leftData, rightData = MCA.pcs(self, n)

		if n is None:
			n = leftData.shape[1]
		
		modes = list(range(1,n+1))
		
		leftPcs = xr.DataArray(leftData,
					  dims 	= ['time','mode'],
					  coords = { 
						  'time' : self.__timeSteps,
						  'mode' : modes
						  })
		
		rightPcs = xr.DataArray(rightData,
					  dims 	= ['time','mode'],
					  coords = {
						  'time' : self.__timeSteps,
						  'mode' : modes
						  })
		
		return leftPcs, rightPcs

	
	def eofs(self, n=None):
		"""Return the first `n` EOFs.

		Parameters
		----------
		n : int, optional
			Number of EOFs to return If none, all EOFs are returned. 
			The default is None.

		Returns
		-------
		DataArray
			EOFs of left input field.
		DataArray
			EOFs of right input field.

		"""
		leftData, rightData = MCA.eofs(self, n)
		
		if n is None:
			n = leftData.shape[-1]
		
		modes = list(range(1,n+1))
		
	
		leftEofs = xr.DataArray(leftData,
					  dims 	= ['lat','lon','mode'],
					  coords = { 
						  'lon' : self.__lonsLeft,
						  'lat' : self.__latsLeft,
						  'mode' : modes
						  })
		
		rightEofs = xr.DataArray(rightData,
					  dims 	= ['lat','lon','mode'],
					  coords = {
						  'lon' : self.__lonsRight,
						  'lat' : self.__latsRight,
						  'mode' : modes
						  })
		
		return leftEofs, rightEofs


	def __getMapBoundaries(self, data):
		assert(isinstance(data, xr.DataArray))
		
		east 	= data.coords['lon'].min()
		west 	= data.coords['lon'].max()
		south 	= data.coords['lat'].min()
		north 	= data.coords['lat'].max()
		
		boundaries = [east, west, south, north]
		return boundaries 


	def __normalizeTo1(self, data):
		return data / abs(data).max()

	def createFigure(self, n=3, grandCols=1):
		rows, cols = [n, 2 * grandCols]
		
		fig 	= plt.figure(figsize = (14 * grandCols, 5 * n))
		gs 		= GridSpec(rows, cols, width_ratios=[1, 1]*grandCols) 
		
		
		axesPC = np.empty((n,grandCols), dtype=mplt.axes.SubplotBase)
		axesEOF = np.empty((n, grandCols), dtype=mplt.axes.SubplotBase)
		
		for i in range(rows):
			for j in range(grandCols):
				axesPC[i,j] = plt.subplot(gs[i,2*j])
				axesEOF[i,j] = plt.subplot(gs[i,2*j+1], projection = ccrs.PlateCarree())
		
		return fig, axesPC, axesEOF
	

	def __validateSigns(self, signs, n):
		"""Check if list of signs match the length n.

		Parameters
		----------
		signs : list
			List of +-1s.
		n : int
			Length to check against.

		Raises
		------
		ValueError
			If `n` does not match the length of `sign`.

		Returns
		-------
		signs : 1-ndarray
			Signs in the correct form.

		"""
		# if nothing provided just take +1 as signs
		if signs is None:
			signs = np.ones(n)
		# otherwise check if signs provided by the user have correct lenght
		else:
			if (n == len(signs)):
				signs = np.array(signs)
			else:
				raise ValueError('Number of PCs and signs need to the same.')	
		return signs		
	
	
	def __flipSigns(self, data, signs):
		modes = data['mode'].size
		signs = self.__validateSigns(signs, modes)
		
		return signs * data


	def plotMode(self, n=1, signs=None, title='', cmap='RdGy_r'):
		"""
		Plot PC and EOF of left data field for mode `n`.

		Parameters
		----------
		n : int, optional
			Mode of PC and EOF to plot. The default is 1.
		signs : list of int, optional
			Either +1 or -1 in order to flip the sign of shown PCs/EOFs. 
			The default is None.
		title : str, optional
			Title of figure. The default is ''.

		Returns
		-------
		None.

		"""
		pcs 		= self.pcs(n)[0].sel(mode=n).real
		eofs 		= self.eofs(n)[0].sel(mode=n).real
		var, varErr = self.explainedVariance(n)
		var, varErr = [var.sel(mode=n).values, varErr.sel(mode=n).values]
		
		# normalize all EOFs such that they range from -1...+1
		eofs 		= self.__normalizeTo1(eofs)

		# flip signs of PCs and EOFs, if needed
		eofs 	= self.__flipSigns(eofs, signs)
		pcs 	= self.__flipSigns(pcs, signs)

		# map boundaries as [east, west, south, north]
		mapBoundaries = self.__getMapBoundaries(eofs)

		fig, axesPC, axesEOF = self.createFigure(1,1)
		axPC = axesPC[0,0]
		axEOF = axesEOF[0,0]
		# plot PCs
		pcs.plot(ax=axPC)
		axPC.set_ylim(-0.06,0.06)
		axPC.set_xlabel('')
		axPC.set_ylabel('PC ' + str(n))
		axPC.set_title('')
		#axPC.text(0.15,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var,varErr)
		#	,transform=axPC.transAxes,horizontalalignment='center')
			
		# plot EOFs
		eofs.plot(ax=axEOF,cmap=cmap,extend='both',
							  add_colorbar=True,vmin=-.8,vmax=.8)
		axEOF.set_extent(mapBoundaries,crs=ccrs.PlateCarree())
		axEOF.coastlines(resolution='50m', lw=0.5)
		axEOF.add_feature(cfeature.LAND.with_scale('50m'))
		
		axEOF.set_title('')
		axEOF.set_aspect('auto')
			
		fig.subplots_adjust(wspace=0.1,hspace=0,left=0.05)
		if title == '':
			title = "PC {} ({:.1f} $\pm$ {:.1f} \%)".format(n, var,varErr)
		fig.suptitle(title, y=1.0)
		
	
	def plotOverview(self, n=3, signs=None, title=''):
		"""
		Plot first `n` PCs of left data field alongside their corresponding EOFs.

		Parameters
		----------
		n : int, optional
			Number of PCs and EOFs to plot. The default is 3.
		signs : list of int, optional
			List of +-1 in order to flip the sign of shown PCs/EOFs. 
			Length of list has to match `n`. The default is None.
		title : str, optional
			Title of figure. The default is ''.

		Returns
		-------
		None.

		"""
		pcs 		= self.pcs(n)[0].real
		eofs 		= self.eofs(n)[0].real
		var, varErr = self.explainedVariance(n)
		var, varErr = [var.values, varErr.values]
		
		# normalize all EOFs such that they range from -1...+1
		eofs 		= self.__normalizeTo1(eofs)

		# flip signs of PCs and EOFs, if needed
		eofs 	= self.__flipSigns(eofs, signs)
		pcs 	= self.__flipSigns(pcs, signs)

		# map boundaries as [east, west, south, north]
		mapBoundaries = self.__getMapBoundaries(eofs)

		fig, axesPC, axesEOF = self.createFigure(n,1)
		
		# plot PCs 
		for i,a in enumerate(axesPC.flatten()):
			pcs.sel(mode=(i+1)).plot(ax=a)
			a.set_xlabel('')
			a.set_ylabel('PC ' + str(i+1))
			a.set_title('')
			a.text(0.1,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var[i],varErr[i])
				,transform=a.transAxes,horizontalalignment='center')
			
		# plot EOFs
		for i,a in enumerate(axesEOF.flatten()):
			eofs.sel(mode=(i+1)).plot(ax=a,cmap='RdGy_r',
								  add_colorbar=False,vmin=-.9,vmax=.9)
			a.coastlines(lw=0.5)
			a.set_extent(mapBoundaries,crs=ccrs.PlateCarree())
			a.set_title('')
			a.set_aspect('auto')
			
		fig.subplots_adjust(wspace=0.1,hspace=0,left=0.05)
		fig.suptitle(title)


	def cplotOverview(self, n=3, signs=None, title=''):
		"""
		Plot first `n` complex PCs of left data field alongside their corresponding EOFs.

		Parameters
		----------
		n : int, optional
			Number of PCs and EOFs to plot. The default is 3.
		signs : list of int, optional
			List of +-1 in order to flip the sign of shown PCs/EOFs. 
			Length of list has to match `n`. The default is None.
		title : str, optional
			Title of figure. The default is ''.

		Returns
		-------
		None.

		"""		
		pcs 		= self.pcs(n)[0]
		eofs 		= self.eofs(n)[0]
		var, varErr = self.explainedVariance(n)
		var, varErr = [var.values, varErr.values]
		
		amplitude = eofs * eofs.conjugate()
		phase = np.arctan2(eofs.imag,eofs.real)
		
		# normalize all EOFs such that they range from -1...+1
		amplitude 	= self.__normalizeTo1(amplitude)

		# flip signs of PCs and EOFs, if needed
		amplitude 	= self.__flipSigns(amplitude, signs)
		pcs 		= self.__flipSigns(pcs, signs)

		# map boundaries as [east, west, south, north]
		mapBoundaries = self.__getMapBoundaries(eofs)

		fig, axesPC, axesEOF = self.createFigure(n,2)
		
		# plot PCs 
		for i,[ar,ac] in enumerate(axesPC):
			pcs.sel(mode=(i+1)).real.plot(ax=ar)
			ar.set_xlabel('')
			ar.set_ylabel('PC ' + str(i+1))
			ar.set_title('')
			ar.text(0.1,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var[i],varErr[i])
				,transform=ar.transAxes,horizontalalignment='center')

			pcs.sel(mode=(i+1)).imag.plot(ax=ac)
			ac.set_xlabel('')
			ac.set_ylabel('PC ' + str(i+1))
			ac.set_title('')
			ac.text(0.1,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var[i],varErr[i])
				,transform=ar.transAxes,horizontalalignment='center')
				
		# plot EOFs
		for i,[ar,ac] in enumerate(axesEOF):
			amplitude.sel(mode=(i+1)).real.plot(ax=ar,cmap='RdGy_r',
								  add_colorbar=False,vmin=-.9,vmax=.9)
			ar.coastlines(lw=0.5)
			ar.set_extent(mapBoundaries,crs=ccrs.PlateCarree())
			ar.set_title('')
			ar.set_aspect('auto')
			
			phase.sel(mode=(i+1)).plot(ax=ac,cmap='twilight',
								  add_colorbar=False,vmin=-3,vmax=3)
			ac.coastlines(lw=0.5)
			ac.set_extent(mapBoundaries,crs=ccrs.PlateCarree())
			ac.set_title('')
			ac.set_aspect('auto')
		
		fig.subplots_adjust(wspace=0.1,hspace=0,left=0.05)
		fig.suptitle(title)



	def plot2Overview(self, n=3, signs=None, title='',cmap='RdGy_r'):
		"""
		Plot first `n` PCs of left and right data field alongside their corresponding EOFs.

		Parameters
		----------
		n : int, optional
			Number of PCs and EOFs to plot. The default is 3.
		signs : list of int, optional
			List of +-1 in order to flip the sign of shown PCs/EOFs. 
			Length of list has to match `n`. The default is None.
		title : str, optional
			Title of figure. The default is ''.

		Returns
		-------
		None.

		"""		
		pcsLeft, pcsRight 	= self.pcs(n)
		eofsLeft, eofsRight = self.eofs(n)
		var, varErr = self.explainedVariance(n)
		var, varErr = [var.values, varErr.values]
		
		# normalize all EOFs such that they range from -1...+1
		eofsLeft 	= self.__normalizeTo1(eofsLeft)
		eofsRight 	= self.__normalizeTo1(eofsRight)

		# flip signs of PCs and EOFs, if needed
		eofsLeft 	= self.__flipSigns(eofsLeft, signs)
		eofsRight 	= self.__flipSigns(eofsRight, signs)
		pcsLeft 	= self.__flipSigns(pcsLeft, signs)
		pcsRight 	= self.__flipSigns(pcsRight, signs)

		# map boundaries as [east, west, south, north]
		mapBoundariesLeft = self.__getMapBoundaries(eofsLeft)
		mapBoundariesRight = self.__getMapBoundaries(eofsRight)

		fig, axesPC, axesEOF = self.createFigure(n,2)
		
		# plot PCs 
		for i,[ar,ac] in enumerate(axesPC):
			pcsLeft.sel(mode=(i+1)).real.plot(ax=ar)
			ar.set_xlabel('')
			ar.set_ylabel('PC ' + str(i+1))
			ar.set_title('')
			ar.text(0.1,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var[i],varErr[i])
				,transform=ar.transAxes,horizontalalignment='center')

			pcsRight.sel(mode=(i+1)).real.plot(ax=ac)
			ac.set_xlabel('')
			ac.set_ylabel('PC ' + str(i+1))
			ac.set_title('')
			ac.text(0.1,0.9,r'{:.1f} $\pm$ {:.1f} \%'.format(var[i],varErr[i])
				,transform=ar.transAxes,horizontalalignment='center')
			
		# plot EOFs
		for i,[ar,ac] in enumerate(axesEOF):
			eofsLeft.sel(mode=(i+1)).real.plot(ax=ar,cmap=cmap,
								  add_colorbar=False,vmin=-.9,vmax=.9)
			ar.coastlines(lw=0.5)
			ar.set_extent(mapBoundariesLeft,crs=ccrs.PlateCarree())
			ar.set_title('')
			ar.set_aspect('auto')
			
			eofsRight.sel(mode=(i+1)).plot(ax=ac,cmap=cmap,
								  add_colorbar=False,vmin=-.9,vmax=.9)
			ac.coastlines(lw=0.5)
			ac.set_extent(mapBoundariesRight,crs=ccrs.PlateCarree())
			ac.set_title('')
			ac.set_aspect('auto')
		
		fig.subplots_adjust(wspace=0.1,hspace=0,left=0.05)
		fig.suptitle(title)
