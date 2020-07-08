#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created by  : Niclas Rieger
# Created on  : Thu Jun  4 22:08:28 2020
# =============================================================================
""" Implementation of VARIMAX and PROMAX rotation. """
# =============================================================================
# Imports
# =============================================================================
import numpy as np

# =============================================================================
# VARIMAX
# =============================================================================

def varimax(A, gamma=1, maxIter=1000, tol=1e-8):
	'''
	Perform (orthogonal) Varimax rotation with Kaiser normalization (Kaiser 1958).

	Parameters
	----------
	A : array-like
		PCs to be rotated. Matrix A has shape n x p with (n: number of time steps; p: number of pcs)
	gamma : float, optional
		Parameter which determines the type of rotation performed: varimax (1), quartimax (0). Other values are possible. The default is 1.
	maxIter : TYPE, optional
		Number of iterations performed. The default is 500.
	tol : TYPE, optional
		Tolerance at which iteration process terminates. The default is 1e-10.

	Returns
	-------
	B : array-like
		Rotated matrix with same dimensions as A.
	R : array-like
		Rotation matrix.

	'''
	A = A.copy()
	n,p = A.shape
    # initialize rotation matrix
	R =	np.eye(p)
	d = 0

    # normalize the matrix
    # using sqrt of the sum of squares (Kaiser)
	h = np.sqrt(np.sum(A**2,axis=1))
	A = np.diag(1./h) @ A

    # seek for rotation matrix based on varimax criteria
	converged = False
	for i in range(maxIter):
		d_old = d
		basis = A @ R

		transformed = A.conjugate().T @ (basis**3 - (gamma/n) *
                             (basis @ np.diag(np.sum(basis**2,axis=0))))

		u,s,vh = np.linalg.svd(transformed)
		R = u @ vh
		d = np.sum(s)
		if abs(d-d_old)/d < tol:
			converged = True
			print('SVD converged successfully!')
			break

	if(not converged):
		print('Error: Rotation process did not converge!')
		A = np.empty(A.shape)
		A[:] = np.nan

    # de-normalize
	A = np.diag(h) @ A

    # perform rotation
	B = A @ R
	return B, R

# =============================================================================
# PROMAX
# =============================================================================

def promax(A, power=1, maxIter=1000, tol=1e-8):
	'''
	Perform (oblique) Promax rotation with Kaiser normalizataion (Kaiser 1958).

	Parameters
	----------
	A : array-like
		PCs to be rotated. Matrix A has shape n x p with (n: number of time steps; p: number of pcs)
	power : int, optional
		Power the Varimax rotated solution to be raised to. If 1, Promax equals Varimax. The default is 1.

	Returns
	-------
	B : array-like
		Rotated matrix with same dimensions as A.
	R : array-like
		Rotation matrix.
	phi : array-like
		Correlation matrix.

	'''
	X = A.copy()
	n, p = X.shape
	if p < 2:
		print('Cannot rotate 1 PC. No rotation performed.')
		return X, np.eye(n), X.conjugate().T @ X

    # first get varimax rotation
	X, R = varimax(X,maxIter=maxIter,tol=tol)

    # pre-normalization by communalities (sum of squared rows)
	h = np.sqrt(np.sum(X**2,axis=1))
	X = np.diag(1./h) @ X    
    
    # "Procustes" equation 
	P = X * np.abs(X)**(power - 1)

    # fit linear regression model of "Procrustes" equation
    # see Richman 1986 for derivation
	L = np.linalg.inv(X.conjugate().T @ X) @ X.conjugate().T @ P

    # calculate diagonal of inverse square
	try:
		sigma_inv = np.diag(np.diag(np.linalg.inv(L.conjugate().T @ L)))
	except np.linalg.LinAlgError:
		sigma_inv = np.diag(np.diag(np.linalg.pinv(L.conjugate().T @ L)))

    # transform and calculate inner products
	L = L @ np.sqrt(sigma_inv)
	B = X @ L

    # post-normalization based on Kaiser
	B =  np.diag(h) @ B

	R = R @ L

	# Correlation matrix
	L_inv = np.linalg.inv(L)
	phi = L_inv @ L_inv.conjugate().T

	return B, R, phi
