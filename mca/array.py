#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex rotated maximum covariance analysis of two numpy arrays.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
from scipy.signal import hilbert

from tools.rotation import promax

# =============================================================================
# MCA
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
		self.__LLeft 	= VLeft @ S
		self.__LRight 	= VRight @ S

		# get PC scores by projecting data fields on loadings
		self.__ULeft 	= self.__noNanDataLeft @ VLeft @ Si
		self.__URight 	= self.__noNanDataRight @ VRight @ Si

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


	def pcs(self, n=None, scaling=0):
		"""Return the first `n` PCs.

		Parameters
		----------
		n : int, optional
			Number of PCs to be returned. The default is None.
		scaling : [0,1], optional
			If 1, scale PCs by square root of eigenvalues. If 0, return
			unscaled PCs. The default is 0.

		Returns
		-------
		pcsLeft : ndarray
			PCs associated with left input field.
		pcsRight : ndarray
			PCs associated with right input field.

		"""
		pcsLeft 	= self.__ULeft[:,:n]
		pcsRight 	= self.__URight[:,:n]

		if (scaling==1):
			pcsLeft 	= pcsLeft * np.sqrt(self.__eigenvalues[:n])
			pcsRight 	= pcsRight * np.sqrt(self.__eigenvalues[:n])

		return pcsLeft, pcsRight


	def eofs(self, n=None, scaling=0):
		"""Return the first `n` EOFs.

		Parameters
		----------
		n : int, optional
			Number of EOFs to be returned. The default is None.
		scaling : [0,1], optional
			If 1, scale PCs by square root of eigenvalues. If 0, return
			unscaled PCs. The default is 0.

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

		if (scaling==1):
			eofsLeft 	= eofsLeft * np.sqrt(self.__eigenvalues[:n])
			eofsRight 	= eofsRight * np.sqrt(self.__eigenvalues[:n])

		return eofsLeft, eofsRight
