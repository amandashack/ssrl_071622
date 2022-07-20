import arpys
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import math
import h5py
from arpys.loaders.maestro import load_maestro_fits_hvscan, align_binding, read_maestro_fits_attrs
# from arpys.loaders.ssrl import load_ssrl_52_photonEscan
from lmfit.models import ThermalDistributionModel, LinearModel, ConstantModel, PolynomialModel
from PyQt5.QtWidgets import QApplication
from scipy.interpolate import RegularGridInterpolator
from pyimagetool import ImageTool

"""
spectrum.attrs - empty dict
spectrum.coords - the coordinate values and names. 'energy': -1.583 - 0.2577, 'slit': -17.9 to 21.7, 
                  and 'photon_energy': 62 - 150
spectrum.data - the 3D array that holds the data
spectrum.dims - the coordinate names/dimensions. 'photon_energy, 'slit', and 'energy'
spectrum['energy'] - 
"""


def load_ssrl_52_photonEscan(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    # 3d dataset, kinetic energy, angle, photon energy
    counts = np.array(f['Data']['Count'])
    I0 = np.abs(np.array(f['MapInfo']['Beamline:I0']))

    xaxis_offsets = np.array(f['MapInfo']['Measurement:XAxis:Offset'])
    xaxis_maxs = np.array(f['MapInfo']['Measurement:XAxis:Maximum'])
    xaxis_size = counts.shape[0]
    try:
        yaxis_offsets = np.array(f['MapInfo']['Measurement:YAxis:Offset'])
        yaxis_deltas = np.array(f['MapInfo']['Measurement:YAxis:Delta'])
        yaxis_size = counts.shape[1]
    except KeyError:
        yaxis_size = counts.shape[1]
        yaxis_offsets = np.repeat(f['Data']['Axes1'].attrs['Offset'], yaxis_size)
        yaxis_deltas = np.repeat(f['Data']['Axes1'].attrs['Delta'], yaxis_size)

    if (type(f['Data']['Axes2'].attrs['Offset']) is str):
        zaxis_coord = f['MapInfo']['Beamline:energy']
        zaxis_size = len(zaxis_coord)
    else:
        zaxis_offset = f['Data']['Axes2'].attrs['Offset']
        zaxis_delta = f['Data']['Axes2'].attrs['Delta']
        zaxis_size = counts.shape[2]
        zaxis_max = zaxis_size * zaxis_delta + zaxis_offset
        zaxis_coord = np.linspace(zaxis_offset, zaxis_max, num=zaxis_size)

    photon_energy_scan_dataarrays = []

    # Slice by slice along z (photon energy)
    for photon_energy_slice in np.arange(zaxis_size):
        ekslice = counts[:, :, photon_energy_slice] / I0[photon_energy_slice]
        kinetic_coords = np.linspace(xaxis_offsets[photon_energy_slice], xaxis_maxs[photon_energy_slice],
                                     num=xaxis_size)
        angle_coords = np.arange(yaxis_size) * yaxis_deltas[photon_energy_slice] + yaxis_offsets[photon_energy_slice]
        dims = ('energy', 'slit')
        coords = {'energy': kinetic_coords, 'slit': angle_coords}
        ekslice_dataarray = xr.DataArray(ekslice, dims=dims, coords=coords)

        # Cut down on window to find ef with initial guess, will always need tuning if mono drifts too much...
        photon_energy = zaxis_coord[photon_energy_slice]
        workfunc = 4.365
        efguess = photon_energy - workfunc
        maxkinetic = np.nanmax(kinetic_coords)
        effinder = ekslice_dataarray.sel({'energy': slice(efguess - 1.0, maxkinetic)})
        ef = effinder.arpes.guess_ef()
        binding_coords = kinetic_coords - ef

        newcoords = {'energy': binding_coords, 'slit': angle_coords}
        ekslice_binding = xr.DataArray(ekslice, dims=dims, coords=newcoords)
        photon_energy_scan_dataarrays.append(ekslice_binding)

    aligned_eks = []
    first_ek = photon_energy_scan_dataarrays[0]
    aligned_eks.append(first_ek)

    for i in np.arange(1, len(photon_energy_scan_dataarrays)):
        interped = photon_energy_scan_dataarrays[i].interp_like(first_ek)
        aligned_eks.append(interped)
    attrs = dict(f['Beamline'].attrs).copy()
    attrs.update(dict(f['Manipulator'].attrs))
    attrs.update(dict(f['Measurement'].attrs))
    attrs.update(dict(f['Temperature'].attrs))
    attrs.update(dict(f['UserSettings'].attrs))
    attrs.update(dict(f['UserSettings']['AnalyserSlit'].attrs))

    aligned_photonE_scan = xr.concat(aligned_eks, 'photon_energy')
    aligned_photonE_scan = aligned_photonE_scan.assign_coords(coords={'photon_energy': zaxis_coord})
    aligned_photonE_scan = aligned_photonE_scan.assign_attrs(attrs=attrs)
    return aligned_photonE_scan


def np_transpose(xar, tr):
    """Transpose the RegularSpacedData
    :param xar: starting xarray
    :param tr: list of the new transposed order
    """
    coords = {}
    dims = []
    for i in tr:
        name = list(xar.coords)[i]
        coords[name] = xar[name].values
        dims.append(list(xar.dims)[i])
    return xr.DataArray(np.transpose(xar.data, tr), dims=dims, coords=coords)


def fix_array(ar, scan_type):
    """
    make your array uniform based on what type of scan it is
    :param ar: input xarray
    :param scan_type: the scan type can be "hv_scan"
    :return: the fixed array
    """
    if scan_type == "hv_scan":
        photon_energy = ar.photon_energy.values
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(photon_energy), len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        print(size_new, size_ar, tr)
        return np_transpose(ar, tr)


def k_forward_normal(ke, theta, v0):
    """
    convert to kx and kz
    :param ke: kinetic energy data
    :param theta: what slit angle to convert at
    :param v0: inner potential
    :return: kx, kz
    """
    rad = math.radians(theta)
    kx = 0.512 * np.sqrt(ke) * np.sin(rad)
    kz = 0.512 * np.sqrt(ke * (np.cos(rad))**2 + v0)
    return kx, kz


def k_forward_normal_partial(ke, theta):
    """
    convert to kx
    :param ke: kinetic energy data
    :param theta: what slit angle to convert at
    :return:
    """
    rad = math.radians(theta)
    kx = 0.512 * np.sqrt(ke) * np.sin(rad)
    return kx


def k_reverse_normal(kx, kz, v0, wf, be):
    """
    convert back to real space
    :param kx:
    :param kz:
    :param v0:
    :param wf:
    :param be:
    :return:
    """
    a = 0.512
    ke = (kz/a)**2 + (kx/a)**2 - v0
    theta = (180/np.pi) * np.arcsin(kx/(a*np.sqrt(ke)))
    hv = ke + wf - be
    return hv, theta


def k_reverse_normal_partial(kx, hv, be, wf):
    """
    convert back to real space
    :param kx:
    :param hv:
    :param be:
    :param wf:
    :return:
    """
    a = 0.512
    ke = hv + be - wf
    theta = (180 / np.pi) * np.arcsin(kx / (a * np.sqrt(ke)))
    return theta


def my_dewarp(spectra, ef_pos):
    """
    dewarping for a given spectra by passing
    in the fermi level curve
    :param spectra:
    :param ef_pos:
    :return:
    """
    #  ef_pos = dewarp(spectra.coords['slit'].values)
    ef_min = np.min(ef_pos)
    ef_max = np.max(ef_pos)
    de = spectra.coords['energy'].values[1] - spectra.coords['energy'].values[0]
    px_to_remove = int(round((ef_max - ef_min) / de))
    dewarped = np.empty((spectra.coords['slit'].size, spectra.coords['energy'].size - px_to_remove))
    for i in range(spectra.coords['slit'].size):
        rm_from_bottom = int(round((ef_pos[i] - ef_min) / de))
        rm_from_top = spectra.coords['energy'].size - (px_to_remove - rm_from_bottom)
        dewarped[i, :] = spectra.values[i, rm_from_bottom:rm_from_top]
    bottom_energy_offset = int(round((ef_max - ef_min) / de))
    energy = spectra.coords['energy'].values[bottom_energy_offset:]
    dw_data = xr.DataArray(dewarped, coords={'energy': energy, 'slit': spectra.coords['slit'].values},
                           dims=['slit', 'energy'], attrs=spectra.attrs)
    dw_new_ef = dw_data['energy'] - ef_max
    dw_data = dw_data.assign_coords({'energy': dw_new_ef})
    return dw_data


def normalize_hvscan(hvscan, x0, x1, e0, e1):
    """
    break hvscan into each photon energy and normalize to integrate intensity
    :param hvscan:
    :param x0: lower slit position
    :param x1: upper slit position
    :param e0: lower energy range
    :param e1: upper energy range
    :return:
    """
    hv_cuts = []
    for hv in hvscan['photon_energy']:
        hv_cut = hvscan.sel({'photon_energy': hv}, method='nearest')
        hv_cut_cr = hv_cut.sel({'energy': slice(e0, e1)}).sel({'slit': slice(x0, x1)})
        sum_thing = hv_cut_cr.sum('slit')
        sum_thing2 = np.sum(sum_thing.values.astype('int64'))
        area = hv_cut_cr.energy.size * hv_cut_cr.slit.size
        average = sum_thing2 / area
        hv_cut_normed = hv_cut / average
        hv_cuts.append(hv_cut_normed)

    hv_scan_normed = xr.concat(hv_cuts, 'photon_energy')
    hv_scan_normed = hv_scan_normed.assign_coords({'photon_energy': hvscan['photon_energy']})
    hv_scan_normed = fix_array(hv_scan_normed, scan_type="hv_scan")
    return hv_scan_normed


def dewarp_2d(spectrum_2d, e0, e1, x1, x2, ds, de, fermi_model, fermi_params, r2_threshold):
    """
    dewarps the spectrum fermi level for a selected photon energy by using the generated
    fermi model. First a list of ef guesses are made and then a dewarp curve is generated using polyfit
    and then sent into my_dewarp.
    :param spectrum_2d: scans.sel({'photon_energy': hv}, method='nearest')
    :param e0:
    :param e1:
    :param x1:
    :param x2:
    :param ds:
    :param de:
    :param fermi_model:
    :param fermi_params:
    :param r2_threshold:
    :return:
    """
    spectrum_2d_crop = spectrum_2d.sel({'energy': slice(e0, e1)}).sel({'slit': slice(x1, x2)})
    spectrum_2d_crop_downsample_slit = spectrum_2d_crop.arpes.downsample({'slit': ds})
    spectrum_2d_crop_downsample = spectrum_2d_crop_downsample_slit.arpes.downsample({'energy': de})
    angle = spectrum_2d['slit']
    angle_downsample = spectrum_2d_crop_downsample['slit']
    ene_downsample = spectrum_2d_crop_downsample['energy']
    init_params = fermi_params
    ef_downsample = []
    ef_sigma = []
    i = 0
    for theta in angle_downsample:
        edc_xr = spectrum_2d_crop_downsample.sel({'slit': theta}, method='nearest')
        edc_vals = edc_xr.values
        fit_result = fermi_model.fit(edc_vals, fermi_params, x=ene_downsample)
        fermi_params = fit_result.params
        ef_value = fermi_params['fermi_center'].value
        ef_error = fermi_params['fermi_center'].stderr
        fit_result_points = fermi_model.eval(fermi_params, x=ene_downsample)
        r2 = 1 - (fit_result.residual.var() / np.var(fit_result_points))
        if r2 < r2_threshold:
            ef_value = np.NaN
            ef_error = np.NaN
            fermi_params = init_params

        else:
            i += 1

        ef_downsample.append(ef_value)
        ef_sigma.append(ef_error)

    if i < 5:
        print('i was less than 5')
        ef_downsample = []
        ef_sigma = []
        for theta in angle_downsample:
            edc_xr = spectrum_2d_crop_downsample.sel({'slit': theta}, method='nearest')
            ef_value = edc_xr.arpes.guess_ef()
            ef_downsample.append(ef_value)
            ef_sigma.append(1)

    aa = np.array(angle_downsample)
    ee = np.array(ef_downsample)
    clean_ef = np.isfinite(aa) & np.isfinite(ee)
    ww = 1.0 / ((np.array(ef_sigma)) ** 2)
    p = np.polyfit(aa[clean_ef], ee[clean_ef], 2, w=ww)
    dw_curve = np.poly1d(p)
    # dw_curve= arpys.Arpes.make_dewarp_curve(aa[clean_ef], ee[clean_ef])
    # modify this to add weights to the fit error bars
    ef = dw_curve(angle.values)
    spectrum_2d_dw = my_dewarp(spectrum_2d, ef)
    return (spectrum_2d_dw, angle_downsample, ef_downsample, ef,
            spectrum_2d_crop_downsample, ef_sigma)


def standardize(ar):
    """
    Makes the spectra intensities between 0 and 1
    :param ar: xarray with data
    :return: xarray standardized to between 0 and 1
    """
    data_values = ar.values
    w_max = np.nanmax(data_values)
    w_min = np.nanmin(data_values)
    nr_values = (ar.values - w_min) / (w_max - w_min)
    slit = ar.slit.values
    energy = ar.energy.values
    photon_energy = ar.photon_energy.values
    n = xr.DataArray(nr_values, dims=('photon_energy', 'slit', 'energy'),
                     coords={'photon_energy': photon_energy, 'slit': slit, 'energy': energy})
    #n = xr.DataArray(nr_values, coords={'photon_energy': photon_energy, 'energy': energy, 'slit': slit},
    #                 dims=['photon_energy', 'energy', 'slit'])
    return n


def generate_fit(edc, window_min, window_max):
    """
    Generate the fit for one EDC given a specified window
    :param edc:
    :param window_min:
    :param window_max:
    :return:
    """
    fermi_func = ThermalDistributionModel(prefix='fermi_', form='fermi')
    params_ = fermi_func.make_params()
    temp = 13.6  # Temperature of map
    k = 8.617333e-5  # Boltzmann in ev/K
    params_['fermi_kt'].set(value=k * temp, vary=True, max=0.03, min=0.001)
    params_['fermi_center'].set(value=0.1, max=0.15, vary=True)
    params_['fermi_amplitude'].set(value=1, vary=False)

    linear_back = LinearModel(prefix='linear_')
    params_.update(linear_back.make_params())
    params_['linear_slope'].set(value=-.1, max=50, vary=False)
    params_['linear_intercept'].set(value=1, vary=True)

    constant = ConstantModel(prefix='constant_')
    params_.update(constant.make_params())
    params_['constant_c'].set(value=0.002, max=100)

    full_model = linear_back * fermi_func + constant
    window = edc.sel({'energy': slice(window_min, window_max)})
    energies = window.energy.values
    # dos = window.values
    init = full_model.eval(params_, x=energies)
    out_ = full_model.fit(init, params_, x=energies)
    params_ = out_.params
    return full_model, out_, params_


def ef_guess_for_edc(scan, e0, e1, dx, ne):
    """
    :param scan:
    :param e0:
    :param e1:
    :param dx:
    :param ne:
    :return:
    """
    ef_guess = []
    for hv in scan['photon_energy']:
        im = scan.sel({'photon_energy': hv}, method='nearest')
        im_cr = im.sel({'energy': slice(e0, e1)}).sel({'slit': slice(-dx, dx)})
        im_cr_ds = im_cr.arpes.downsample({'energy': ne})

        edc = im_cr_ds.sum('slit')
        ef_est = edc.arpes.guess_ef()
        ef_guess.append(ef_est)
    return ef_guess


def find_spacing(mm, hv, ke, slit, inner_potential=14, num_x=None, num_z=None, partial=False):
    if partial:
        if num_x is None:
            middle_ke = int(len(ke) / 2)
            middle_slit = int(len(slit) / 2)
            kx_ur = k_forward_normal_partial(ke[middle_ke], slit[middle_slit])
            kx_ll = k_forward_normal_partial(ke[middle_ke - 1], slit[middle_slit - 1])
            dkx = kx_ur - kx_ll
            num_x = abs(int((mm[1] - mm[0]) / dkx))
            print(num_x, len(slit))
            return num_x
        else:
            return num_x
    elif not partial:
        if num_x is None and num_z is None:
            middle_ke = int(len(ke) / 2)
            middle_slit = int(len(slit) / 2)
            kx_ur, kz_ur = k_forward_normal(ke[middle_ke],
                                            slit[middle_slit], inner_potential)
            kx_ll, kz_ll = k_forward_normal(ke[middle_ke - 1],
                                            slit[middle_slit - 1], inner_potential)
            dkx = kx_ur - kx_ll
            dkz = kz_ur - kz_ll
            print("\n\n\nThese are the spacing: ", dkx, dkz)
            num_x = abs(int((mm[0][1] - mm[0][0]) / dkx))
            num_z = abs(int((mm[1][1] - mm[1][0]) / dkz))
            print(num_x, num_z, len(slit), len(hv))
        elif num_x or num_z:
            if num_x is None:
                num_x = slit.size
            elif num_z is None:
                num_z = hv.size
        else:
            print("There is something wrong in choosing your "
                  "spacing for reg grid kz conversion")
    return num_x, num_z


def convert_2d_normal_emission(scans, be=0, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param be: binding energy to select to look at a kz cut
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    slit = energy_iso['slit'].values
    ke = photon_energy_iso - wf + be
    print("\nlsdjfpsodjf: ", scans.attrs, "\nfinally youre here: ", scans.coords,
          "\nthis is the data: ", scans.data, "\nthis is the dims: ",
          scans.dims, "\nphoton energy values: ", energy_iso['photon_energy'].values,
          '\nbinding energy 0: ', energy_iso, '\nke: ', ke, '\nslit: ', slit)
    interp_object = RegularGridInterpolator((photon_energy_iso, slit),
                                            energy_iso.values, bounds_error=False)
    kx_ur, kz_ur = k_forward_normal(np.nanmax(ke),
                                    np.nanmax(slit), inner_potential)
    kx_ul, kz_ul = k_forward_normal(np.nanmax(ke),
                                    np.nanmin(slit), inner_potential)
    kx_um, kz_um = k_forward_normal(np.nanmax(ke),
                                    0, inner_potential)
    kx_lr, kz_lr = k_forward_normal(np.nanmin(ke),
                                    np.nanmax(slit), inner_potential)
    kx_ll, kz_ll = k_forward_normal(np.nanmin(ke),
                                    np.nanmin(slit), inner_potential)
    xx = [kx_ur, kx_ul, kx_um, kx_lr, kx_ll]
    zz = [kz_ur, kz_ul, kz_um, kz_lr, kz_ll]
    min_x = min(xx)
    max_x = max(xx)
    min_z = min(zz)
    max_z = max(zz)
    mm = [[min_x, max_x], [min_z, max_z]]
    num_x, num_z = find_spacing(mm, photon_energy_iso, ke, slit, inner_potential, num_x, num_z, partial=False)
    print('\n\nnumber spacing: ', num_x, num_z, slit.size, photon_energy_iso.size)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    kz_new = np.sort(np.linspace(min_z, max_z, num=num_z, endpoint=True))
    kxx, kzz = np.meshgrid(kx_new, kz_new, indexing='ij', sparse=False)
    hv, theta = k_reverse_normal(kxx, kzz, inner_potential, wf, be)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], hv.shape[1]), order='C')
    return xr.DataArray(interp_out, dims=['kx', 'kz'], coords={'kx': kx_new, 'kz': kz_new})


def convert_3d_normal_emission(scans, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    binding_energy = scans.energy.values
    photon_energy = scans.photon_energy.values
    slit = scans.slit.values
    max_ke1 = np.nanmax(photon_energy) - wf + np.nanmax(binding_energy)
    max_ke2 = np.nanmin(photon_energy) - wf + np.nanmax(binding_energy)
    min_ke = np.nanmin(photon_energy) - wf + np.nanmin(binding_energy)
    print("\nfinally youre here: ", scans.coords,
          "\nthis is the data: ", scans, "\nthis is the dims: ",
          scans.dims, "\nbinding energy values: ", binding_energy.shape, '\nslit: ', slit.shape,
          "\nscan shape: ", scans.shape, "\nphoton energy: ", photon_energy.shape)
    interp_object = RegularGridInterpolator((photon_energy, slit, binding_energy),
                                            scans.data, bounds_error=False)
    kx_ur, kz_ur = k_forward_normal(max_ke1,
                                    np.nanmax(slit), inner_potential)
    kx_ul, kz_ul = k_forward_normal(max_ke1,
                                    np.nanmin(slit), inner_potential)
    kx_um, kz_um = k_forward_normal(max_ke1,
                                    0, inner_potential)
    kx_lr_maxke, kz_lr_maxke = k_forward_normal(max_ke2,
                                                np.nanmax(slit), inner_potential)
    kx_ll_maxke, kz_ll_maxke = k_forward_normal(max_ke2,
                                                np.nanmin(slit), inner_potential)
    kx_lr_minke, kz_lr_minke = k_forward_normal(min_ke,
                                                np.nanmax(slit), inner_potential)
    kx_ll_minke, kz_ll_minke = k_forward_normal(min_ke,
                                                np.nanmin(slit), inner_potential)
    xx = [kx_ur, kx_ul, kx_um, kx_lr_maxke, kx_ll_maxke, kx_lr_minke, kx_ll_minke]
    zz = [kz_ur, kz_ul, kz_um, kz_lr_maxke, kz_ll_maxke, kz_lr_minke, kz_ll_minke]
    min_x = min(xx)
    max_x = max(xx)
    min_z = min(zz)
    max_z = max(zz)
    mm = [[min_x, max_x], [min_z, max_z]]
    be = np.nanmax(binding_energy)
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    ke_iso = photon_energy_iso - wf - be
    num_x, num_z = find_spacing(mm, photon_energy_iso, ke_iso, slit, inner_potential, num_x, num_z, partial=False)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    kz_new = np.sort(np.linspace(min_z, max_z, num=num_z, endpoint=True))
    be_new = np.sort(np.linspace(np.nanmin(binding_energy), np.nanmax(binding_energy),
                                 num=len(binding_energy), endpoint=True))
    kxx, kzz, be_grid = np.meshgrid(kx_new, kz_new, be_new, indexing='ij', sparse=False)
    hv, theta = k_reverse_normal(kxx, kzz, inner_potential, wf, be_grid)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C'),
                              be_grid.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], theta.shape[1], be_grid.shape[2]), order='C')
    return xr.DataArray(interp_out, dims=['kx', 'kz', 'energy'], coords={'kx': kx_new,
                                                                         'kz': kz_new,
                                                                         'energy': be_new})


def convert_partial_3d_normal_emission(scans, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    binding_energy = scans.energy.values
    photon_energy = scans.photon_energy.values
    slit = scans.slit.values
    max_ke1 = np.nanmax(photon_energy) - wf + np.nanmax(binding_energy)
    max_ke2 = np.nanmin(photon_energy) - wf + np.nanmax(binding_energy)
    min_ke = np.nanmin(photon_energy) - wf + np.nanmin(binding_energy)
    interp_object = RegularGridInterpolator((photon_energy, slit, binding_energy),
                                            scans.data, bounds_error=False)
    kx_ur = k_forward_normal_partial(max_ke1, np.nanmax(slit))
    kx_ul = k_forward_normal_partial(max_ke1, np.nanmin(slit))
    kx_um = k_forward_normal_partial(max_ke1, 0)
    kx_lr_maxke = k_forward_normal_partial(max_ke2, np.nanmax(slit))
    kx_ll_maxke = k_forward_normal_partial(max_ke2, np.nanmin(slit))
    kx_lr_minke = k_forward_normal_partial(min_ke, np.nanmax(slit))
    kx_ll_minke = k_forward_normal_partial(min_ke, np.nanmin(slit))
    xx = [kx_ur, kx_ul, kx_um, kx_lr_maxke, kx_ll_maxke, kx_lr_minke, kx_ll_minke]
    min_x = min(xx)
    max_x = max(xx)
    mm = [min_x, max_x]
    be = np.nanmax(binding_energy)
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    ke_iso = photon_energy_iso - wf - be
    num_x = find_spacing(mm, photon_energy_iso, ke_iso, slit, inner_potential, num_x, num_z, partial=True)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    hv, kxx, be = np.meshgrid(photon_energy, kx_new, binding_energy,
                              indexing='ij', sparse=False)
    theta = k_reverse_normal_partial(kxx, hv, be, wf)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C'),
                               be.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], theta.shape[1], be.shape[2]), order='C')
    return xr.DataArray(interp_out, dims=['photon_energy', 'kx', 'energy'],
                        coords={'photon_energy': photon_energy,
                                'kx': kx_new,
                                'energy': binding_energy})


def run_dewarp(scans, ef_guess, m, p):
    hv_scans_dewarped = []
    i = 0
    for hv in scans['photon_energy']:
        ef_ini = ef_guess[i]
        p['fermi_center'].set(value=ef_ini, max=0.15, vary=True)
        im_2d = scans.sel({'photon_energy': hv}, method='nearest')
        im_2d_dw = dewarp_2d(im_2d, ef_ini - 0.12, ef_ini + 0.2, -12, 12, 50, 2, m, p, 0.95)
        hv_scans_dewarped.append(im_2d_dw[0])
        i += 1
    hv_dewarp_interp = [hv_scans_dewarped[0]]
    for scan_no in np.arange(1, len(hv_scans_dewarped)):
        hv_dewarp_interp.append(hv_scans_dewarped[scan_no].interp_like(hv_scans_dewarped[0]))
    hv_out = xr.concat(hv_dewarp_interp, 'photon_energy')
    hv_out = hv_out.assign_coords({'photon_energy': scans['photon_energy']})
    hv_out = normalize_hvscan(hv_out, -10, 10, -0.3, 0.1)
    fix_array(hv_out, scan_type='hv_scan')
    return hv_out


def run_edc_plot(edc):
    fig1, ax1 = plt.subplots()
    edc.plot(x='energy', ax=ax1)
    ax1.axvline(-5e-2, color='black')
    fig1.patch.set_facecolor('white')
    fig1.patch.set_alpha(0.95)
    plt.show()


def run_plot(cp):
    app = QApplication([])
    imtool = cp.arpes.plot()
    app.exec_()


def run_2d_plot(pl):
    figa, axa = plt.subplots()
    pl.plot(x='kx', y='kz', ax=axa, add_colorbar=False)
    axa.set_title("")
    axa.set_xlabel('Photon Energy (eV)')
    axa.set_ylabel('Binding Energy (eV)')
    figa.set_size_inches(15, 9)
    plt.show()


if __name__ == "__main__":
    """
    The order of operations:
    1. load in the data
    2. find gamma and shift it to zero if necessary
    ----- make the fermi edge straight -----
    3. standardize the data so that all values fall between 0 and 1
    4. look at the data and choose a photon energy range to look at
    where the binding energy window is large enough to fit a function 
    to the fermi edge
    5. use guess_ef to get a starting place for dewarping
    6. for each photon energy, dewarp the spectra by seitting the initi
    """

    path = "C:/Users/amsh/Documents/coding/arpes/data/ssrl_071522"
    for file in os.listdir(path):
        if file.endswith("17.h5"):
            path1 = os.path.join(path, file)
        elif file.endswith('18.h5'):
            path2 = os.path.join(path, file)
        elif file.endswith('36.h5'):
            path3 = os.path.join(path, file)

    hv_scan_11 = load_ssrl_52_photonEscan(path1)
    hv_scan_16 = load_ssrl_52_photonEscan(path2)
    hv_scan_36 = load_ssrl_52_photonEscan(path3)
    # this number is obtained by determining the offset from
    # zero of the gamma point, normal emission should correspond to 0 deg
    hv_scan_11 = fix_array(hv_scan_11, scan_type='hv_scan')
    normed = standardize(hv_scan_11) #hv_scan_13)
    #run_plot(normed)
    # cut = normed.sel({'photon_energy': 40}, method='nearest')
    # edc_sum_1 = cut.sel({'energy': slice(-0.07, 0.3)}).sum('slit')
    # edc_sum = edc_sum_1.values / (len(edc_sum_1.values))
    # edc = xr.DataArray(edc_sum, coords={'energy': edc_sum_1['energy'].values}, dims=['energy'])
    # run_edc_plot(edc)
    # model, out, params = generate_fit(edc, -0.07, 0.3)
    # I use this to control the photon energy range used for dewarping
    # dd = normed.sel({'photon_energy': slice(30, 48)})
    # run_plot(dd)
    # e0 = -0.2
    # e1 = 0.2
    # dx = 14
    # ne = 2
    # ef_guess = ef_guess_for_edc(dd, e0, e1, dx, ne)
    # hv_dw = run_dewarp(dd, ef_guess, model, params)
    # run_plot(hv_dw)
    full_kz = convert_partial_3d_normal_emission(normed)
    #full_kz = convert_3d_normal_emission(normed)#, num_x=700, num_z=700)
    # run_2d_plot(full_kz.sel({"energy": 0}, method='nearest'))
    run_plot(full_kz)
