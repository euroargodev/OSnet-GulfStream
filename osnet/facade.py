import os
from joblib import load as jbload
import glob
from tensorflow import keras
from abc import ABC
import numpy as np
import gsw
import xarray as xr
from numba import guvectorize
import pkg_resources
import logging

from .utilities import check_and_complement


log = logging.getLogger("osnet.facade")


class predictor_proto(ABC):
    adjust_mld = True
    info = {'name': '?', 'ref': '?', 'models': '?'}

    def __repr__(self):
        summary = ["<osnet.%s>" % self.info['name']]
        summary.append("Reference: %s" % self.info['ref'])
        summary.append("Models: %s" % self.info['models'])
        summary.append("MLD adjusted: %s" % self.adjust_mld)
        return "\n".join(summary)

    def summary(self, id=0):
        """ Show keras network summary for a model instance """
        return self.models[id].summary()

    def _sign_predictions(self, this_obj):
        """ Add attribute to 'sign' a variable as associated with OSnet """
        if isinstance(this_obj, xr.Dataset):
            this_obj.attrs['OSnet'] = "This dataset has variable(s) generated with OSnet-%s model: %s" % (self.info['name'], self.info['ref'])
        if isinstance(this_obj, xr.DataArray):
            this_obj.attrs['OSnet'] = "This variable has been generated with OSnet-%s model: %s" % (self.info['name'], self.info['ref'])
        return this_obj

    def _is_signed(self, this_obj):
        return 'OSnet' in this_obj.attrs

    @property
    def SDL(self):
        """ Standard Depth Levels of predictions """
        return [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55,
                        60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
                        301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]

    def _mask_X(self, x):
        """ Compute a mask to make the X vector full, i.e. without NaN """
        M = list()
        Features = ['SST', 'SLA', 'MDT', 'BATHY']
        for v in Features:
            M.append(x[v].notnull())
        mask = xr.concat(M, dim='n_features')
        mask = mask.sum(dim='n_features')
        mask = mask == len(Features)
        return mask

    def _make_X(self, x):
        """ create X vector """
        # Stack and mask in the input array:
        x = x.stack({'sampling': list(x.dims)})
        self._mask = self._mask_X(x)
        x = x.where(self._mask == 1, drop=True)

        # Create [Samples, Features] array to work with:
        N_samples = len(x['sampling'])
        X = np.zeros([N_samples, 12])
        log.debug("Osnet working with X[%i,%i] input array" % (X.shape[0], X.shape[1]))
        X[:, 0] = x['SLA'].data
        X[:, 1] = x['lat'].data
        X[:, 2] = x['lon'].data
        X[:, 3] = np.cos(np.pi * 2 * 1 / 365 * x['dayOfYear'].data)
        X[:, 4] = np.sin(np.pi * 2 * 1 / 365 * x['dayOfYear'].data)
        X[:, 5] = x['MDT'].data
        X[:, 6] = x['UGOSA'].data
        X[:, 7] = x['VGOSA'].data
        X[:, 8] = x['UGOS'].data
        X[:, 9] = x['VGOS'].data
        X[:, 10] = x['SST'].data
        X[:, 11] = -np.abs(x['BATHY'].data)
        return X, x

    def _get_mean_std_pred(self, ensemble, X, Sm, Sstd, Tm, Tstd):
        predS = []
        predT = []
        mld = []
        for model in ensemble:
            tmp_pred = model.predict(X)
            temp = tmp_pred[:, :, 1] * Tstd + Tm
            psal = tmp_pred[:, :, 0] * Sstd + Sm
            predS.append(psal)
            predT.append(temp)
            mld.append(tmp_pred[:, :, 2])
        return np.mean(predS, axis=0), np.std(predS, axis=0), np.mean(predT, axis=0), np.std(predT, axis=0), np.mean(
            mld, axis=0)

    def _get_MLD_from_mask(self, mask):
        depth_levels = self.SDL
        mask = np.sign(0.5 - mask)
        return depth_levels[np.argmin(mask)]

    def _add_sig(self, ds):
        SA = gsw.SA_from_SP(ds['psal'], ds['PRES_INTERPOLATED'], ds['lon'], ds['lat'])
        CT = gsw.CT_from_t(SA, ds['temp'], ds['PRES_INTERPOLATED'])
        sig = gsw.sigma0(SA, CT)
        return ds.assign(variables={"sig": sig})

    def _add_maskv3(self, ds):
        b = 2
        b2 = 1
        H = 0.5664  # For OSnet Gulf Stream, see Pauthenet et al, 2022
        mask2 = np.where(ds['MLD_mask'].data < H, ds['MLD_mask'], 1)
        ds = ds.assign(variables={"MLD_mask2": (('sampling', 'PRES_INTERPOLATED'), mask2)})
        mask3 = np.where((ds['MLD_mask'] > H) & (ds['MLD_mask'] < b2), b - ds['MLD_mask'].data, ds['MLD_mask2'].data)
        ds = ds.assign(variables={"MLD_mask3": (('sampling', 'PRES_INTERPOLATED'), mask3)})
        return ds

    def _post_processing_adjustment(self, ds, mask):
        @guvectorize(
            "(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
            "(n), (n), (n), (n) -> (n), (n)"
        )
        def apply_mask_1d(temp_in, psal_in, depth, mask, temp, psal):
            temp[:] = np.copy(temp_in)
            psal[:] = np.copy(psal_in)
            for d in range(len(depth) - 2, -1, -1):
                # apply mask on TEMP and PSAL
                temp[d] = (temp_in[d] * mask[d] - temp_in[d + 1] * mask[d]) + temp[d + 1]
                psal[d] = (psal_in[d] * mask[d] - psal_in[d + 1] * mask[d]) + psal[d + 1]

        temp_out, psal_out = xr.apply_ufunc(apply_mask_1d,
                                            ds['temp'], ds['psal'], ds['PRES_INTERPOLATED'], mask,
                                            input_core_dims=(
                                            ['PRES_INTERPOLATED'], ['PRES_INTERPOLATED'], ['PRES_INTERPOLATED'],
                                            ['PRES_INTERPOLATED']),
                                            output_core_dims=(['PRES_INTERPOLATED'], ['PRES_INTERPOLATED']),
                                            output_dtypes=[np.float64, np.float64])
        # get sig adjusted
        sa_out = gsw.SA_from_SP(psal_out, ds['PRES_INTERPOLATED'], ds['lon'], ds['lat'])
        ct_out = gsw.CT_from_t(sa_out, temp_out, ds['PRES_INTERPOLATED'])
        sig_out = gsw.sigma0(sa_out, ct_out)

        ds_out = ds.assign(variables={"temp_adj": (('sampling', 'PRES_INTERPOLATED'), temp_out.data),
                                      "psal_adj": (('sampling', 'PRES_INTERPOLATED'), psal_out.data),
                                      "sig_adj": (('sampling', 'PRES_INTERPOLATED'), sig_out.data)})
        return ds_out

    def _predict(self, x, ensemble, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd, scaler_input, **kwargs):
        """ Make prediction """
        X, y = self._make_X(x)  # y is stacked/masked x
        X_scaled = scaler_input.transform(X)

        # Prediction:
        get_MLD_from_mask_vect = np.vectorize(self._get_MLD_from_mask, signature='(k)->()')
        pred_S_mean, pred_S_std, pred_T_mean, pred_T_std, mld = self._get_mean_std_pred(ensemble, X_scaled, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd)

        # Add to dataset:
        y = y.assign(variables={"PRES_INTERPOLATED": (("PRES_INTERPOLATED"), self.SDL)})
        y = y.assign(variables={"temp": (("sampling", "PRES_INTERPOLATED"), pred_T_mean)})
        y = y.assign(variables={"temp_std": (("sampling", "PRES_INTERPOLATED"), pred_T_std)})
        y = y.assign(variables={"psal": (("sampling", "PRES_INTERPOLATED"), pred_S_mean)})
        y = y.assign(variables={"psal_std": (("sampling", "PRES_INTERPOLATED"), pred_S_std)})
        y = y.assign(variables={"mld": (("sampling"), get_MLD_from_mask_vect(mld.data))})
        vlist = ['temp', 'temp_std', 'psal', 'psal_std', 'mld']

        # Adjust Mixed layer:
        if ('adjust_mld' in kwargs and kwargs['adjust_mld']) or self.adjust_mld:
            y = y.assign(variables={"MLD_mask": (("sampling", "PRES_INTERPOLATED"), mld.data)})
            y = self._add_sig(y)
            y = self._add_maskv3(y)
            y = self._post_processing_adjustment(y, mask=y['MLD_mask3'])
            y = y.drop_vars(['MLD_mask', 'MLD_mask2', 'MLD_mask3'])
            vlist += ['sig', 'temp_adj', 'psal_adj', 'sig_adj']

        # # Add attributes:
        # for v in vlist:
        #     y[v] = self._sign_predictions(y[v])
        #
        # # Prepare output according to input mask
        # x = x.stack({'sampling': list(x.dims)})
        # for v in list(set(y.data_vars) - set(x.data_vars)):
        #     x = x.assign({v: y[v]})
        # x = x.unstack('sampling')
        #
        # if (np.prod(x['SST'].shape) != self._xmask.shape[0]):
        #     log.debug("Unravelled data not matching mask dimension, re-indexing")
        #     mask = self._xmask.unstack()
        #     x['SST'] = x['SST'].reindex_like(mask)

        return y.unstack('sampling')


class predictor(predictor_proto):
    def __init__(self,
                 name='Gulf-Stream',
                 adjust_mld=True):
        if name == 'Gulf-Stream':
            self.default_path = pkg_resources.resource_filename("osnet", os.path.sep.join(["models", "models_Gulf_Stream"]))
            self._load()
            self.info['name'] = 'GulfStream'
            self.info['ref'] = 'Pauthenet et al, 2022 (http://dx.doi.org/...)'
            self.info['models'] = '%i instance(s) in the ensemble' % (len(self.models))
        else:
            raise ValueError('Unknown model name')
        self.adjust_mld = adjust_mld

    def _load(self, path=None):
        if path is None:
            path = self.default_path

        # Load scalers:
        self.scalers = {}
        self.scalers['scaler_input'] = jbload(os.path.sep.join([path, "scaler_input.joblib"]))
        self.scalers['scal_Sm'] = jbload(os.path.sep.join([path, "Sm.joblib"]))
        self.scalers['scal_Sstd'] = jbload(os.path.sep.join([path, "Sstd.joblib"]))
        self.scalers['scal_Tm'] = jbload(os.path.sep.join([path, "Tm.joblib"]))
        self.scalers['scal_Tstd'] = jbload(os.path.sep.join([path, "Tstd.joblib"]))

        # Load models
        models_list = glob.glob(os.path.sep.join([path, "neuralnet", "ensemble", "*"]))
        ensemble = []
        for model_path in models_list:
            ensemble.append(keras.models.load_model(model_path, compile=False))
        self.models = ensemble

        if path is not None:
            self.info['name'] = 'custom'
            self.info['ref'] = '-'
            self.info['models'] = '%i instance(s) in the ensemble' % (len(self.models))

        return self

    def predict(self, x, inplace=True, keep_added=False, **kwargs):
        """ Make T/S/MLD predictions

        Parameters
        ----------
        x: :class:`xarray.DataSet`
            Input dataset.
            Must have at least the following coordinates:
                - 'lon', 'lat', 'time'
            Otherwise, we expect:
                1. ``dayOfYear``
                1. ``BATHY``
                1. ``lat`` in the range: [18.125, 54.875]
                1. ``lon`` in the range: [275.125, 334.875] or [-84.875, -25.125]
                1. ``SST``
                1. ``SLA``
                1. ``UGOSA``
                1. ``VGOSA``
                1. ``MDT``
                1. ``UGOS``
                1. ``VGOS``
        inplace: bool (True)
            Should predictions be added to the input dataset (Default) or not.

        Returns
        -------
        y: :class:`xarray.DataSet`
            Dataset with OSnet predictions of temperature, salinity profiles and Mixed Layer Depth
        """
        if self._is_signed(x):
            raise ValueError("Cannot make predictions from a dataset that already has OSnet variables")
        else:
            x = check_and_complement(x)

        y = self._predict(x=x,
                            ensemble=self.models,
                            scal_Sm=self.scalers['scal_Sm'], scal_Sstd=self.scalers['scal_Sstd'],
                            scal_Tm=self.scalers['scal_Tm'], scal_Tstd=self.scalers['scal_Tstd'],
                            scaler_input=self.scalers['scaler_input'],
                            suffix="_TS",
                            **kwargs)

        # ds_output has ds_inputs variables dimensions broadcasted, so we need to re-assign to their original shape:
        for v in x.data_vars:
            if v in y:
                y = y.assign({v: x[v]})

        # Return dataset:
        if inplace:
            # Add predicted fields to input dataset:
            # log.debug("INPLACE: Add predicted fields to input dataset:")
            # log.debug(list(set(y.data_vars) - set(x.data_vars)))
            for v in list(set(y.data_vars) - set(x.data_vars)):
                x = x.assign({v: y[v]})
            out = x
        else:
            # Remove input arrays from the predicted dataset:
            # log.debug("NOT INPLACE: Remove input arrays from the predicted dataset:")
            # log.debug(list(set(x.data_vars)))
            y = y.drop_vars(list(set(x.data_vars)))
            out = y

        # Possibly remove data we had to add to the input:
        if 'OSnet-added' in out.attrs and not keep_added:
            out = out.drop_vars(out.attrs['OSnet-added'].split(";"))

        return self._sign_predictions(out)
