#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complex rotated maximum covariance analysis of two xarray DataArrays.
"""

# =============================================================================
# Imports
# =============================================================================
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mca.array import MCA

# =============================================================================
# xMCA
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

    def __init__(self, left, right=None, normalize=True, coslat=False):
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
        self._left  = left.copy()

        # for univariate case (normal PCA) left = right
        self._right = self._getRightField(right)

        assert(self._isXarray(self._left))
        assert(self._isXarray(self._right))

        # store meta information of time steps and coordinates
        self._timeSteps 	= self._left.coords['time'].values
        self._lonsLeft 	    = self._left.coords['lon'].values
        self._lonsRight 	= self._right.coords['lon'].values
        self._latsLeft 	    = self._left.coords['lat'].values
        self._latsRight 	= self._right.coords['lat'].values
        self._nameLeft      = self._getFieldName(self._left,'left')
        self._nameRight     = self._getFieldName(self._right,'right')

        self._left     = self._centerXarray(self._left)
        self._right    = self._centerXarray(self._right)

        self.normalize = normalize
        if self.normalize:
            self._left     = self._normalizeXarray(self._left)
            self._right    = self._normalizeXarray(self._right)


        if coslat:
            # coslat correction needs to happen AFTER normalization since
            # normalization mathematically removes the coslat correction effect
            self._left     = self._applyCosLatCorrection(self._left)
            self._right    = self._applyCosLatCorrection(self._right)
            # Deactivate normalization for array.MCA, otherwise
            # coslat correction may be overwritten
            self.normalize = False

        dataLeft 	= self._left.data
        dataRight 	= self._right.data

        # constructor of base class for np.ndarray
        MCA.__init__(self, dataLeft, dataRight, self.normalize)


    def _isXarray(self, data):
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

    def _getFieldName(self, dataArray, backupName='undefined'):
        name = dataArray.name
        if name is None:
            name = backupName
        return name

    def _centerXarray(self, xarray):
        return xarray - xarray.mean('time')


    def _normalizeXarray(self, xarray):
        return xarray / xarray.std('time')


    def _applyCosLatCorrection(self, data):
        """Apply area correction to higher latitudes.

        """

        weights     = np.cos(np.deg2rad(data.lat))

        return data * weights


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
            name 	= 'error-eigenvalues')

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
            name 	= 'explained-variance')
        error = xr.DataArray(desVarErr,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'error-explained-variance')

        return values, error


    def pcs(self, n=None, scaling=0):
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
        leftData, rightData = MCA.pcs(self, n, scaling=scaling)

        if n is None:
            n = leftData.shape[1]

        modes = list(range(1,n+1))

        leftPcs = xr.DataArray(leftData,
          dims 	= ['time','mode'],
          coords = {
          'time' : self._timeSteps,
          'mode' : modes
          },
          name = '-'.join([self._nameLeft,'pcs']))

        rightPcs = xr.DataArray(rightData,
          dims 	= ['time','mode'],
          coords = {
          'time' : self._timeSteps,
          'mode' : modes
          },
          name = '-'.join([self._nameRight,'pcs']))

        return leftPcs, rightPcs


    def eofs(self, n=None, scaling=0):
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
        leftData, rightData = MCA.eofs(self, n, scaling=scaling)

        if n is None:
            n = leftData.shape[-1]

        modes = list(range(1,n+1))


        leftEofs = xr.DataArray(leftData,
          dims 	= ['lat','lon','mode'],
          coords = {
          'lon' : self._lonsLeft,
          'lat' : self._latsLeft,
          'mode' : modes
          },
          name = '-'.join([self._nameLeft,'eofs']))

        rightEofs = xr.DataArray(rightData,
          dims 	= ['lat','lon','mode'],
          coords = {
          'lon' : self._lonsRight,
          'lat' : self._latsRight,
          'mode' : modes
          },
          name = '-'.join([self._nameRight,'eofs']))

        return leftEofs, rightEofs


    def spatialAmplitude(self, n=None):
        """Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        eofsLeft, eofsRight = self.eofs(n)

        amplitudeLeft   = np.sqrt(eofsLeft * eofsLeft.conjugate())
        amplitudeRight  = np.sqrt(eofsRight * eofsRight.conjugate())

        amplitudeLeft.name  = '-'.join([self._nameLeft,'spatial-amplitude'])
        amplitudeRight.name = '-'.join([self._nameRight,'spatial-amplitude'])

        # use the real part to force a real output
        return amplitudeLeft.real, amplitudeRight.real


    def spatialPhase(self, n=None):
        """Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        eofsLeft, eofsRight = self.eofs(n)

        phaseLeft = np.arctan2(eofsLeft.imag,eofsLeft.real)
        phaseRight = np.arctan2(eofsRight.imag,eofsRight.real)

        phaseLeft.name  = '-'.join([self._nameLeft,'spatial-phase'])
        phaseRight.name = '-'.join([self._nameRight,'spatial-phase'])

        # use the real part to force a real output
        return phaseLeft.real, phaseRight.real

    def temporalAmplitude(self, n=None):
        """Return the temporal amplitude functions for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude functions to return. If none, all functions are returned.
            The default is None.

        Returns
        -------
        DataArray
            Temporal amplitude function of left input field.
        DataArray
            Temporal amplitude function of right input field.

        """
        pcsLeft, pcsRight = self.pcs(n)

        amplitudeLeft   = np.sqrt(pcsLeft * pcsLeft.conjugate())
        amplitudeRight  = np.sqrt(pcsRight * pcsRight.conjugate())

        amplitudeLeft.name  = '-'.join([self._nameLeft,'temporal-amplitude'])
        amplitudeRight.name = '-'.join([self._nameRight,'temporal-amplitude'])

        # use the real part to force a real output
        return amplitudeLeft.real, amplitudeRight.real


    def temporalPhase(self, n=None):
        """Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, all functions are returned.
            The default is None.

        Returns
        -------
        DataArray
            Temporal phase function of left input field.
        DataArray
            Temporal phase function of right input field.

        """
        pcsLeft, pcsRight = self.pcs(n)

        phaseLeft = np.arctan2(pcsLeft.imag,pcsLeft.real)
        phaseRight = np.arctan2(pcsRight.imag,pcsRight.real)

        phaseLeft.name  = '-'.join([self._nameLeft,'temporal-phase'])
        phaseRight.name = '-'.join([self._nameRight,'temporal-phase'])

        # use the real part to force a real output
        return phaseLeft.real, phaseRight.real


    def _getMapBoundaries(self, data):
        assert(isinstance(data, xr.DataArray))

        east 	= data.coords['lon'].min()
        west 	= data.coords['lon'].max()
        south 	= data.coords['lat'].min()
        north 	= data.coords['lat'].max()

        boundaries = [east, west, south, north]
        return boundaries


    def _normalizeEOFto1(self, data):
        return data / abs(data).max(['lon','lat'])


    def _normalizePCto1(self, data):
        return data / abs(data).max(['time'])


    def _createFigure(self, nrows=3, coltypes=['t','s'], cenLon=0):
        nRows, nCols = [nrows, len(coltypes)]

        # positions of temporal plots
        isTemporalCol 	= [True if i=='t' else False for i in coltypes]

        # set projections associated with temporal/spatial plots
        projTemporalPlot 	= None
        projSpatialPlot 	= ccrs.PlateCarree(central_longitude=cenLon)
        projections = [projTemporalPlot if i=='t' else projSpatialPlot for i in coltypes]

        # set relative width of temporal/spatial plots
        widthTemporalPlot 	= 4
        widthSpatialPlot 	= 5
        widths = [widthTemporalPlot if i=='t' else widthSpatialPlot for i in coltypes]

        # create figure environment
        fig 	= plt.figure(figsize = (7 * nCols, 5 * nRows))
        gs 		= GridSpec(nRows, nCols, width_ratios=widths)
        axes 	= np.empty((nRows, nCols), dtype=mpl.axes.SubplotBase)

        for i in range(nRows):
            for j,proj in enumerate(projections):
                axes[i,j] = plt.subplot(gs[i,j], projection=proj)

        axesPC = axes[:,isTemporalCol]
        axesEOF = axes[:,np.logical_not(isTemporalCol)]

        return fig, axesPC, axesEOF


    def _validateSigns(self, signs, n):
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


    def _flipSigns(self, data, signs):
        modes = data['mode'].size
        signs = self._validateSigns(signs, modes)

        return signs * data


    def _calculateCorrelation(self, x, y):
        assert(self._isXarray(x))
        assert(self._isXarray(y))

        x = x - x.mean('time')
        y = y - y.mean('time')

        xy = (x*y).mean('time')
        sigx = x.std('time')
        sigy = y.std('time')

        return xy/sigx/sigy


    def homogeneousPatterns(self, n=None):
        """
        Return left and right homogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all patterns
            are returned. The default is None.

        Returns
        -------
        xr.DataArray
            Left homogeneous correlation maps.
        xr.DataArray
            Right homogeneous correlation maps.

        """

        pcsLeft, pcsRight 		= self.pcs(n)
        pcsLeft, pcsRight 		= [pcsLeft.real, pcsRight.real]

        fieldLeft  = self._left
        fieldRight = self._right

        homPatternsLeft 	= self._calculateCorrelation(fieldLeft,pcsLeft)
        homPatternsRight 	= self._calculateCorrelation(fieldRight,pcsRight)

        return homPatternsLeft, homPatternsRight


    def heterogeneousPatterns(self, n=None):
        """
        Return left and right heterogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all patterns
            are returned. The default is None.

        Returns
        -------
        xr.DataArray
            Left heterogeneous correlation maps.
        xr.DataArray
            Right heterogeneous correlation maps.

        """
        pcsLeft, pcsRight 		= self.pcs(n)
        pcsLeft, pcsRight 		= [pcsLeft.real, pcsRight.real]

        fieldLeft  = self._left
        fieldRight = self._right

        hetPatternsLeft 	= self._calculateCorrelation(fieldLeft,pcsRight)
        hetPatternsRight 	= self._calculateCorrelation(fieldRight,pcsLeft)

        return hetPatternsLeft, hetPatternsRight


    def plotMode(self, n=1, right=False, signs=None, title='', cmap='RdGy_r'):
        """
        Plot mode`n` PC and EOF of left (and right) data field.

        Parameters
        ----------
        n : int, optional
            Mode of PC and EOF to plot. The default is 1.
        right : boolean
            Plot PC and EOF of right field. The default is False.
        signs : list of int, optional
            Either +1 or -1 in order to flip the sign of shown PCs/EOFs.
            The default is None.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        pcsLeft, pcsRight 		= self.pcs(n)
        pcsLeft, pcsRight 		= [pcsLeft.sel(mode=n).real, pcsRight.sel(mode=n).real]

        eofsLeft, eofsRight 	= self.eofs(n)
        eofsLeft, eofsRight 	= [eofsLeft.sel(mode=n).real, eofsRight.sel(mode=n).real]

        var, varErr 			= self.explainedVariance(n)
        var, varErr 			= [var.sel(mode=n).values, varErr.sel(mode=n).values]


        # normalize all EOFs/PCs such that they range from -1...+1
        eofsLeft 		= self._normalizeEOFto1(eofsLeft)
        eofsRight 		= self._normalizeEOFto1(eofsRight)
        pcsLeft 		= self._normalizePCto1(pcsLeft)
        pcsRight 		= self._normalizePCto1(pcsRight)

        # flip signs of PCs and EOFs, if needed
        eofsLeft 	= self._flipSigns(eofsLeft, signs)
        eofsRight 	= self._flipSigns(eofsRight, signs)
        pcsLeft 	= self._flipSigns(pcsLeft, signs)
        pcsRight 	= self._flipSigns(pcsRight, signs)

        # map boundaries as [east, west, south, north]
        mapBoundariesLeft  = self._getMapBoundaries(eofsLeft)
        mapBoundariesRight = self._getMapBoundaries(eofsRight)

        # mapProjection and center longitude for
        mapProjection = ccrs.PlateCarree()
        cenLon  = int((mapBoundariesLeft[0] + mapBoundariesLeft[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        if right:
            fig, axesPC, axesEOF = self._createFigure(2,['t','s'],cenLon)
        else:
            fig, axesPC, axesEOF = self._createFigure(1,['t','s'],cenLon)



        # plot PCs/EOFs
        pcsLeft.plot(ax = axesPC[0,0])
        eofsLeft.plot(
            ax = axesEOF[0,0], transform = mapProjection, cmap = cmap,
            extend = 'neither',	add_colorbar = True, vmin = -1, vmax = 1,
            cbar_kwargs = {'label': 'EOF (normalized)'})
        axesEOF[0,0].set_extent(mapBoundariesLeft, crs = mapProjection)

        if right:
            pcsRight.plot(ax = axesPC[1,0])
            eofsRight.plot(
                ax=axesEOF[1,0], transform = mapProjection, cmap = cmap,
                extend = 'neither', add_colorbar = True, vmin = -1, vmax = 1,
                cbar_kwargs = {'label': 'EOF (normalized)'})
            axesEOF[1,0].set_extent(mapBoundariesRight, crs = mapProjection)


        for i,a in enumerate(axesPC[:,0]):
            a.set_ylim(-1,1)
            a.set_xlabel('')
            a.set_ylabel('PC (normalized)')
            a.set_title('')


        for i,a in enumerate(axesEOF[:,0]):
            a.coastlines(resolution='50m', lw=0.5)
            a.add_feature(cfeature.LAND.with_scale('50m'))
            a.set_title('')
            a.set_aspect('auto')


        fig.subplots_adjust(wspace=0.1,hspace=0.2,left=0.05)

        if title == '':
            title = "PC {} ({:.1f} $\pm$ {:.1f} \%)".format(n, var,varErr)

        if right:
            yOffset = 0.95
        else:
            yOffset = 1.00
        fig.suptitle(title, y=yOffset)


    def cplotMode(self, n=1, right=False, threshold=0, title='', cmap='pink_r'):
        """
        Plot mode`n` PC and EOF of left (and right) data field.

        Parameters
        ----------
        n : int, optional
            Mode of PC and EOF to plot. The default is 1.
        right : boolean
            Plot PC and EOF of right field. The default is False.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        pcsLeft, pcsRight 	= self.pcs(n)
        pcsLeft, pcsRight 	= [pcsLeft.sel(mode=n).real, pcsRight.sel(mode=n).real]

        amplitudeLeft, amplitudeRight   = self.spatialAmplitude(n)
        amplitudeLeft, amplitudeRight   = [amplitudeLeft.sel(mode=n), amplitudeRight.sel(mode=n)]

        phaseLeft, phaseRight           = self.spatialPhase(n)
        phaseLeft, phaseRight           = [phaseLeft.sel(mode=n),phaseRight.sel(mode=n)]

        var, varErr 		= self.explainedVariance(n)
        var, varErr 		= [var.sel(mode=n).values, varErr.sel(mode=n).values]


        # normalize all EOFs/PCs such that they range from -1...+1
        amplitudeLeft   = self._normalizeEOFto1(amplitudeLeft)
        amplitudeRight  = self._normalizeEOFto1(amplitudeRight)
        pcsLeft         = self._normalizePCto1(pcsLeft)
        pcsRight        = self._normalizePCto1(pcsRight)

        # apply amplitude threshold
        amplitudeLeft   = amplitudeLeft.where(amplitudeLeft > threshold)
        amplitudeRight  = amplitudeRight.where(amplitudeRight > threshold)
        phaseLeft       = phaseLeft.where(amplitudeLeft > threshold)
        phaseRight      = phaseRight.where(amplitudeRight > threshold)

        # map boundaries as [east, west, south, north]
        mapBoundariesLeft  = self._getMapBoundaries(amplitudeLeft)
        mapBoundariesRight = self._getMapBoundaries(amplitudeRight)

        # mapProjection and center longitude for
        mapProjection = ccrs.PlateCarree()
        cenLon  = int((mapBoundariesLeft[0] + mapBoundariesLeft[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        # create figure environment
        if right:
            fig, axesPC, axesEOF = self._createFigure(2,['t','s','s'], cenLon)
        else:
            fig, axesPC, axesEOF = self._createFigure(1,['t','s','s'], cenLon)



        # plot PCs/Amplitude/Phase
        pcsLeft.real.plot(ax = axesPC[0,0])
        amplitudeLeft.real.plot(
            ax = axesEOF[0,0], transform = mapProjection,
            cmap = cmap, extend = 'neither', add_colorbar = True,
            vmin = 0, vmax = 1, cbar_kwargs = {'label' : 'Amplitude (normalized)'})
        phaseLeft.plot(
            ax = axesEOF[0,1], transform = mapProjection,
            cmap = 'twilight_shifted', cbar_kwargs = {'label' : 'Phase (rad)'},
            add_colorbar = True, vmin = -np.pi, vmax = np.pi)

        axesEOF[0,0].set_extent(mapBoundariesLeft,crs = mapProjection)
        axesEOF[0,1].set_extent(mapBoundariesLeft,crs = mapProjection)

        axesEOF[0,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,varErr))
        axesEOF[0,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,varErr))


        if right:
            pcsRight.real.plot(ax = axesPC[1,0])
            amplitudeRight.real.plot(
                ax = axesEOF[1,0], transform = mapProjection,
                cmap = cmap, extend = 'neither', add_colorbar = True, vmin = 0,
                vmax = 1, cbar_kwargs = {'label' : 'Amplitude (normalized)'})
            phaseRight.plot(
                ax = axesEOF[1,1], transform = mapProjection,
                cmap = 'twilight_shifted', cbar_kwargs = {'label' : 'Phase (rad)'},
                add_colorbar = True, vmin = -np.pi, vmax = 	np.pi)

            axesEOF[1,0].set_extent(mapBoundariesRight,crs = mapProjection)
            axesEOF[1,1].set_extent(mapBoundariesRight,crs = mapProjection)

            axesEOF[1,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,varErr))
            axesEOF[1,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,varErr))

        for a in axesPC.flatten():
            a.set_ylabel('Real PC (normalized)')
            a.set_xlabel('')
            a.set_title('')


        for a in axesEOF.flatten():
            a.coastlines(lw = 0.5, resolution = '50m')
            a.set_aspect('auto')

        fig.subplots_adjust(wspace = 0.1, hspace = 0.17, left = 0.05)
        fig.suptitle(title)


    def plotOverview(self, n=3, right=False, signs=None, title='', cmap='RdGy_r'):
        """
        Plot first `n` PCs and EOFs of left data field.

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
        pcsLeft, pcsRight 		= self.pcs(n)
        pcsLeft, pcsRight 		= [pcsLeft.real, pcsRight.real]

        eofsLeft, eofsRight 	= self.eofs(n)
        eofsLeft, eofsRight 	= [eofsLeft.real, eofsRight.real]

        var, varErr 			= self.explainedVariance(n)
        var, varErr 			= [var.values, varErr.values]


        # normalize all EOFs/PCs such that they range from -1...+1
        eofsLeft 		= self._normalizeEOFto1(eofsLeft)
        eofsRight 		= self._normalizeEOFto1(eofsRight)
        pcsLeft 		= self._normalizePCto1(pcsLeft)
        pcsRight 		= self._normalizePCto1(pcsRight)

        # flip signs of PCs and EOFs, if needed
        eofsLeft 	= self._flipSigns(eofsLeft, signs)
        eofsRight 	= self._flipSigns(eofsRight, signs)
        pcsLeft 	= self._flipSigns(pcsLeft, signs)
        pcsRight 	= self._flipSigns(pcsRight, signs)

        # map boundaries as [east, west, south, north]
        mapBoundariesLeft = self._getMapBoundaries(eofsLeft)
        mapBoundariesRight = self._getMapBoundaries(eofsRight)

        # mapProjection and center longitude for
        mapProjection = ccrs.PlateCarree()
        cenLon  = int((mapBoundariesLeft[0] + mapBoundariesLeft[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        if right:
            fig, axesPC, axesEOF = self._createFigure(n,['t','s','s','t'], cenLon)
        else:
            fig, axesPC, axesEOF = self._createFigure(n,['t','s'], cenLon)


        # plot PCs/EOFs
        for i in range(n):
            pcsLeft.sel(mode = (i+1)).plot(ax = axesPC[i,0])
            eofsLeft.sel(mode = (i+1)).plot(
                ax = axesEOF[i,0],
                transform = mapProjection, cmap = cmap, extend = 'neither',
                add_colorbar = True, vmin = -1,	vmax = 1,
                cbar_kwargs = {'label' : 'EOF (normalized)'})
            axesEOF[i,0].set_extent(mapBoundariesLeft,crs = mapProjection)
            axesEOF[i,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))

        if right:
            for i in range(n):
                pcsRight.sel(mode = (i+1)).plot(ax = axesPC[i,1])
                eofsRight.sel(mode = (i+1)).plot(
                    ax = axesEOF[i,1],
                    transform = mapProjection, cmap = cmap, extend = 'neither',
                    add_colorbar = True, vmin = -1, vmax = 1,
                    cbar_kwargs = {'label': 'EOF (normalized)'})
                axesEOF[i,1].set_extent(mapBoundariesRight,crs = mapProjection)
                axesEOF[i,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))


        for a in axesPC.flatten():
            a.set_ylim(-1,1)
            a.set_xlabel('')
            a.set_ylabel('PC (normalized)')
            a.set_title('')

        if right:
            for a in axesPC[:,1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")

        # plot EOFs
        for a in axesEOF.flatten():
            a.coastlines(lw = 0.5)
            a.set_aspect('auto')

        fig.subplots_adjust(wspace = .1, hspace = 0.2, left = 0.05)
        fig.suptitle(title)


    def cplotOverview(self, n=3, right=False, threshold=0, title='', cmap='pink_r'):
        """
        Plot first `n` complex PCs of left data field alongside their corresponding EOFs.

        Parameters
        ----------
        n : int, optional
            Number of PCs and EOFs to plot. The default is 3.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        pcsLeft, pcsRight 		= self.pcs(n)
        pcsLeft, pcsRight 		= [pcsLeft.real, pcsRight.real]

        amplitudeLeft, amplitudeRight   = self.spatialAmplitude(n)
        phaseLeft, phaseRight           = self.spatialPhase(n)

        var, varErr 			= self.explainedVariance(n)
        var, varErr 			= [var.values, varErr.values]


        # normalize all EOFs/PCs such that they range from -1...+1
        amplitudeLeft 		= self._normalizeEOFto1(amplitudeLeft)
        amplitudeRight 		= self._normalizeEOFto1(amplitudeRight)
        pcsLeft 		= self._normalizePCto1(pcsLeft)
        pcsRight 		= self._normalizePCto1(pcsRight)

        # apply amplitude threshold
        amplitudeLeft   = amplitudeLeft.where(amplitudeLeft > threshold)
        amplitudeRight  = amplitudeRight.where(amplitudeRight > threshold)
        phaseLeft       = phaseLeft.where(amplitudeLeft > threshold)
        phaseRight      = phaseRight.where(amplitudeRight > threshold)


        # map boundaries as [east, west, south, north]
        mapBoundariesLeft = self._getMapBoundaries(amplitudeLeft)
        mapBoundariesRight = self._getMapBoundaries(amplitudeRight)

        # mapProjection and center longitude for
        mapProjection = ccrs.PlateCarree()
        cenLon  = int((mapBoundariesLeft[0] + mapBoundariesLeft[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        # create figure environment
        if right:
            fig, axesPC, axesEOF = self._createFigure(n,['t','s','s','s','s','t'], cenLon)
        else:
            fig, axesPC, axesEOF = self._createFigure(n,['t','s','s'], cenLon)



        # plot PCs/Amplitude/Phase
        for i in range(n):
            pcsLeft.sel(mode=(i+1)).real.plot(ax = axesPC[i,0])

            amplitudeLeft.sel(mode=(i+1)).real.plot(
                ax = axesEOF[i,0],
                transform = mapProjection, cmap = cmap, extend = 'neither',
                add_colorbar = True, vmin = 0, vmax = 1,
                cbar_kwargs = {'label' : 'Amplitude (normalized)'})

            phaseLeft.sel(mode=(i+1)).plot(
                ax = axesEOF[i,1],
                transform = mapProjection, cmap = 'twilight_shifted',
                cbar_kwargs = {'label' : 'Phase (rad)'}, add_colorbar = True,
                vmin = -np.pi, vmax = np.pi)

            axesEOF[i,0].set_extent(mapBoundariesLeft,crs=mapProjection)
            axesEOF[i,1].set_extent(mapBoundariesLeft,crs=mapProjection)

            axesEOF[i,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))
            axesEOF[i,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))

        if right:
            for i in range(n):
                pcsRight.sel(mode=(i+1)).real.plot(ax = axesPC[i,1])

                amplitudeRight.sel(mode=(i+1)).real.plot(
                    ax = axesEOF[i,2],
                    transform = mapProjection, cmap = cmap, extend = 'neither',
                    add_colorbar = True, vmin = 0, vmax = 1,
                    cbar_kwargs = {'label' : 'Amplitude (normalized)'})

                phaseRight.sel(mode=(i+1)).plot(
                    ax = axesEOF[i,3],
                    transform = mapProjection, cmap = 'twilight_shifted',
                    cbar_kwargs = {'label': 'Phase (rad)'}, add_colorbar = True,
                    vmin = -np.pi, vmax = np.pi)

                axesEOF[i,2].set_extent(mapBoundariesRight,crs=ccrs.PlateCarree())
                axesEOF[i,3].set_extent(mapBoundariesRight,crs=ccrs.PlateCarree())

                axesEOF[i,2].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))
                axesEOF[i,3].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],varErr[i]))

        if right:
            for a in axesPC[:,1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")

        for a in axesPC.flatten():
            a.set_ylabel('Real PC (normalized)')
            a.set_xlabel('')
            a.set_title('')


        for a in axesEOF.flatten():
            a.coastlines(lw = 0.5, resolution = '50m')
            a.set_aspect('auto')


        fig.subplots_adjust(wspace = 0.1, hspace = 0.17, left = 0.05)
        fig.suptitle(title)


    def splitComplex(self, data_array):
        ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
        return ds

    def toNetcdf(self, dataArray, path, *args, **kwargs):
        fileName    = '.'.join([dataArray.name,'nc'])
        powerIdx    = 'p{:02}'.format(self._power)
        rotIdx      = 'r{:02}'.format(self._nRotations)
        complexIdx  = 'c{:}'.format(int(self._useHilbert))
        if self._useMCA:
            methodIdx = 'mca'
        else:
            methodIdx = 'pca'

        fileName    = '-'.join([methodIdx,complexIdx,rotIdx,powerIdx,fileName])
        finalPath   = os.path.join(path,fileName)

        if self._useHilbert:
            dataSet = self.splitComplex(dataArray)
        else:
            dataSet = dataArray

        dataSet.to_netcdf(path=finalPath, *args, **kwargs)

    def saveAnalysis(self, path=None):
        if self._useMCA:
            analysisName = '-'.join([self._nameLeft,self._nameRight])
        else:
            analysisName = self._nameLeft

        if path is None:
            path = os.getcwd()
        outputDirectory = os.path.join(path, 'output')
        analysisDirectory = os.path.join(outputDirectory,analysisName)

        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)
        if not os.path.exists(analysisDirectory):
            os.makedirs(analysisDirectory)

        self.toNetcdf(self.eofs()[0],analysisDirectory)
        self.toNetcdf(self.pcs()[0],analysisDirectory)
        self.toNetcdf(self.eigenvalues()[0],analysisDirectory)
        self.toNetcdf(self.explainedVariance()[0],analysisDirectory)
        self.toNetcdf(self.spatialAmplitude()[0],analysisDirectory)
        self.toNetcdf(self.spatialPhase()[0],analysisDirectory)
        self.toNetcdf(self.temporalAmplitude()[0],analysisDirectory)
        self.toNetcdf(self.temporalPhase()[0],analysisDirectory)

        if self._useMCA:
            self.toNetcdf(self.eofs()[1],analysisDirectory)
            self.toNetcdf(self.pcs()[1],analysisDirectory)
            self.toNetcdf(self.spatialAmplitude()[1],analysisDirectory)
            self.toNetcdf(self.spatialPhase()[1],analysisDirectory)
            self.toNetcdf(self.temporalAmplitude()[1],analysisDirectory)
            self.toNetcdf(self.temporalPhase()[1],analysisDirectory)
