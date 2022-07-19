import arpys
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
from arpys.loaders.ssrl import load_ssrl_52
from PyQt5.QtWidgets import QApplication


def be_to_ke(hv, spectra):
    workfunc = 4.5
    be = np.array(spectra.energy)
    ke = hv - workfunc + be
    reassigned = spectra.assign_coords({'energy': ke})
    return reassigned


def ke_to_be(hv, spectra):
    wf = 4.5
    ke = np.array(spectra.energy)
    be = ke - hv + wf
    reassigned = spectra.assign_coords({'energy': be})
    return reassigned


def run_plot(cp):
    app = QApplication([])
    imtool = cp.arpes.plot()
    app.exec_()


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
        if file.endswith("21.h5"):
            path1 = os.path.join(path, file)
        elif file.endswith('26.h5'):
            path2 = os.path.join(path, file)
        elif file.endswith('23.h5'):
            path3 = os.path.join(path, file)

    hv_scan_21 = load_ssrl_52(path1)
    hv_scan_26 = load_ssrl_52(path2)
    hv_scan_23 = load_ssrl_52(path3)
    photon_energy_21 = 76
    photon_energy_26 = 92
    photon_energy_23 = 115
    hv_scan_21 = ke_to_be(photon_energy_21, hv_scan_21)
    hv_scan_26 = ke_to_be(photon_energy_26, hv_scan_26)
    hv_scan_23 = ke_to_be(photon_energy_23, hv_scan_23)
    run_plot(hv_scan_23)

