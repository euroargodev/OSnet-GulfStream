from joblib import load as jbload
import glob
from tensorflow import keras
from abc import ABC, abstractmethod
import numpy as np
import gsw
import xarray as xr
from numba import float64, guvectorize

class osnet_proto(ABC):
    adjust_mld = True

    def _make_X(self, x):
        """ create X vector """
        d = 1/365
        cos_week = np.cos(np.pi * 2 * d * x['dayOfYear'].data)
        sin_week = np.sin(np.pi * 2 * d * x['dayOfYear'].data)
        X = np.zeros([len(x['lat']), 12])
        X[:,0] = x['SLA'].data
        X[:,1] = x['lat'].data
        X[:,2] = x['lon'].data
        X[:,3] = cos_week
        X[:,4] = sin_week
        X[:,5] = x['MDT'].data
        X[:,6] = x['UGOSA'].data
        X[:,7] = x['VGOSA'].data
        X[:,8] = x['UGOS'].data
        X[:,9] = x['VGOS'].data
        X[:,10] = x['SST'].data
        X[:,11] = -x['BATHY'].data
        return X

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
        depth_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55,
                        60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
                        301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]
        mask = np.sign(0.5 - mask)
        return depth_levels[np.argmin(mask)]

    def _add_sig(self, ds):
        SA = gsw.SA_from_SP(ds['psal'], ds['PRES_INTERPOLATED'], ds['lon'], ds['lat'])
        CT = gsw.CT_from_t(SA, ds['temp'], ds['PRES_INTERPOLATED'])
        sig = gsw.sigma0(SA, CT)
        ds = ds.assign(variables={"sig": (('lat', 'PRES_INTERPOLATED'), sig.data)})
        return ds

    def _add_maskv3(self, ds):
        b = 2
        b2 = 1
        H = 0.5664  # For OSnet Gulf Stream, see Pauthenet et al, 2022
        mask2 = np.where(ds['MLD_mask'].data < H, ds['MLD_mask'], 1)
        ds = ds.assign(variables={"MLD_mask2": (('lat', 'PRES_INTERPOLATED'), mask2)})
        mask3 = np.where((ds['MLD_mask'] > H) & (ds['MLD_mask'] < b2), b - ds['MLD_mask'].data, ds['MLD_mask2'].data)
        ds = ds.assign(variables={"MLD_mask3": (('lat', 'PRES_INTERPOLATED'), mask3)})
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

        ds_out = ds.assign(variables={"temp_adj": (('lat', 'PRES_INTERPOLATED'), temp_out.data),
                                      "psal_adj": (('lat', 'PRES_INTERPOLATED'), psal_out.data),
                                      "sig_adj": (('lat', 'PRES_INTERPOLATED'), sig_out.data)})
        return ds_out

    def _predict(self, x, ensemble, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd, scaler_input, **kwargs):
        """ Make prediction """
        X = self._make_X(x)
        X_scaled = scaler_input.transform(X)

        # ------------- Predict and add to dataset -------------- #
        get_MLD_from_mask_vect = np.vectorize(self._get_MLD_from_mask, signature='(k)->()')
        pred_S_mean, pred_S_std, pred_T_mean, pred_T_std, mld = self._get_mean_std_pred(ensemble, X_scaled, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd)
        x = x.assign(variables={"psal": (('lat', 'PRES_INTERPOLATED'), pred_S_mean.data)})
        x = x.assign(variables={"temp": (('lat', 'PRES_INTERPOLATED'), pred_T_mean.data)})
        x = x.assign(variables={"MLD_mask": (('lat', 'PRES_INTERPOLATED'), mld.data)})
        x = x.assign(variables={"mld": (('lat'), get_MLD_from_mask_vect(x.MLD_mask))})
        x = x.assign(variables={"psal_std": (('lat', 'PRES_INTERPOLATED'), pred_S_std.data)})
        x = x.assign(variables={"temp_std": (('lat', 'PRES_INTERPOLATED'), pred_T_std.data)})

        if ('adjust_mld' in kwargs and kwargs['adjust_mld']) or self.adjust_mld:
            x = self._add_sig(x)
            x = self._add_maskv3(x)
            x = self._post_processing_adjustment(x, mask=x['MLD_mask3'])

        return x

class osnet(osnet_proto):
    def __init__(self,
                 name='Gulf-Stream',
                 adjust_mld=True):
        if name == 'Gulf-Stream':
            self.model_path = 'models_Gulf_Stream'
            self.load()
        else:
            raise ValueError('Unknown model name')
        self.adjust_mld = adjust_mld

    def load(self, path=None):
        if path is None:
            path = self.model_path

        # Load scalers:
        self.scalers = {}
        self.scalers['scaler_input'] = jbload(f"{path}/scaler_input.joblib")
        self.scalers['scal_Sm'] = jbload(f'{path}/Sm.joblib')
        self.scalers['scal_Sstd'] = jbload(f'{path}/Sstd.joblib')
        self.scalers['scal_Tm'] = jbload(f'{path}/Tm.joblib')
        self.scalers['scal_Tstd'] = jbload(f'{path}/Tstd.joblib')

        # Load models
        models_list = glob.glob(f'{path}/neuralnet/ensemble/*')
        ensemble = []
        for model_path in models_list:
            ensemble.append(keras.models.load_model(model_path, compile=False))
        self.models = ensemble
        return self

    def predict(self, ds_inputs, **kwargs):
        return self._predict(x=ds_inputs,
                             ensemble=self.models,
                             scal_Sm=self.scalers['scal_Sm'], scal_Sstd=self.scalers['scal_Sstd'],
                             scal_Tm=self.scalers['scal_Tm'], scal_Tstd=self.scalers['scal_Tstd'],
                             scaler_input=self.scalers['scaler_input'],
                             suffix="_TS",
                             **kwargs)