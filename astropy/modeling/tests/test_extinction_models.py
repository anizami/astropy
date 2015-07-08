# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for extinction curve."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from .. import extinction_models as ext
from ..extinction_models import *
from ... import units as u
import pytest

extinction_models = ['ccm89', 'od94', 'gcc09', 'f99', 'fm07', 'wd01', 'd03']


def test_extinction_shapes():
    pytest.importorskip("scipy")
    for model in extinction_models:

        # single value should work
        extinction(1.e4 * u.angstrom, a_v=1., model=model)

        # multiple values should return appropriate shape
        assert extinction([1.e4] * u.angstrom, a_v=1., model=model).shape == (1,)
        assert extinction([1.e4, 2.e4] * u.angstrom, a_v=1., model=model).shape == (2,)

# TODO: This is a TODO from the specutils package
# Test is only to precision of 0.015 because there is a discrepancy
# of 0.014 for the B band wavelength of unknown origin (and up to 0.002 in
# other bands).
#
# Note that a and b can be obtained with:
# b = extinction_ccm89(wave, ebv=1., r_v=0.)
# a = extinction_ccm89(wave, ebv=1., r_v=1.) - b
#
# These differ from the values tablulated in the original paper.
# Could be due to floating point errors in the original paper?


def test_extinction_ccm89():

    # U, B, V, R, I, J, H, K band effective wavelengths from CCM '89 table 3
    x = np.array([2.78, 2.27, 1.82, 1.43, 1.11, 0.80, 0.63, 0.46]) * (1 / u.micron)

    # A(lambda)/A(V) for R_V = 3.1 from Table 3 of CCM '89
    ratio_true = np.array([1.569, 1.337, 1.000, 0.751, 0.479, 0.282, 0.190,
                           0.114])

    wave = 1 / x  # wavelengths in Angstroms

    model = Extinction_ccm89(a_v=1., r_v=3.1)
    a_lambda_over_a_v = model(wave).value

    # a_lambda_over_a_v = extinction_ccm89(wave, a_v=1., r_v=3.1)
    # assert (u.Quantity(a_lambda_over_a_v, unit=u.dimensionless_unscaled).unit ==
    #         u.dimensionless_unscaled)
    np.testing.assert_allclose(a_lambda_over_a_v, ratio_true, atol=0.015)


# TODO: The tabulated values go to 0.001, but the test is only for matching
# at the 0.005 level, because there is currently a discrepancy up to 0.0047
# of unknown origin.

def test_extinction_ccm89_nd():
    wave = np.random.uniform(5000, 6000, (100, 100, 100)) * u.angstrom

    model = Extinction_ccm89(a_v=1., r_v=3.1)
    calculated_extinction = model(wave).value
    assert calculated_extinction.shape == (100, 100, 100)


def test_extinction_od94():
    """
    Tests the broadband extinction estimates from O'Donnell 1998
    at Rv = 3.1 against the widely used values tabulated in
    Schlegel, Finkbeiner and Davis (1998)
    http://adsabs.harvard.edu/abs/1998ApJ...500..525S

    This is tested by evaluating the extinction curve at a (given)
    effective wavelength, since these effective wavelengths:
    "... represent(s) that wavelength on the extinction curve
    with the same extinction as the full passband."

    The test does not include UKIRT L' (which, at 3.8 microns) is
    beyond the range of wavelengths currently in specutils
    or the APM b_J filter which is defined in a non-standard way.

    Precision is tested to the significance of the SFD98
    tabulated values (1e-3).
    """
    sfd_eff_waves = np.array([3372., 4404., 5428., 6509., 8090., 3683., 4393.,
                              5519., 6602., 8046., 12660., 16732., 22152.,
                              5244., 6707., 7985., 9055., 6993., 3502., 4676.,
                              4127., 4861., 5479., 3546., 4925., 6335., 7799.,
                              9294., 3047., 4711., 5498., 6042., 7068., 8066.,
                              4814., 6571., 8183.]) * u.Angstrom
    sfd_filter_names = np.array([
            'Landolt_U', 'Landolt_B', 'Landolt_V', 'Landolt_R', 'Landolt_I',
            'CTIO_U', 'CTIO_B', 'CTIO_V', 'CTIO_R', 'CTIO_I', 'UKIRT_J',
            'UKIRT_H', 'UKIRT_K', 'Gunn_g', 'Gunn_r', 'Gunn_i', 'Gunn_z',
            'Spinard_R', 'Stromgren_u', 'Stromgren_b', 'Stromgren_v',
            'Stromgren_beta', 'Stromgren_y', 'Sloan_u', 'Sloan_g', 'Sloan_r',
            'Sloan_i', 'Sloan_z', 'WFPC2_F300W', 'WFPC2_F450W', 'WFPC2_F555W',
            'WFPC2_F606W', 'WFPC2_F702W', 'WFPC2_F814W', 'DSSII_g', 'DSSII_r',
            'DSSII_i'])
    sfd_table_alambda = np.array([1.664, 1.321, 1.015, 0.819, 0.594, 1.521,
                                  1.324, 0.992, 0.807, 0.601, 0.276, 0.176,
                                  0.112, 1.065, 0.793, 0.610, 0.472, 0.755,
                                  1.602, 1.240, 1.394, 1.182, 1.004, 1.579,
                                  1.161, 0.843, 0.639, 0.453, 1.791, 1.229,
                                  0.996, 0.885, 0.746, 0.597, 1.197, 0.811,
                                  0.580])
    model = Extinction_od94(a_v=1., r_v=3.1)
    od94_alambda = model(sfd_eff_waves).value
    np.testing.assert_allclose(sfd_table_alambda, od94_alambda, atol=0.005)


def test_extinction_fm07():
    pytest.importorskip('scipy')
    wave = np.arange(3000, 9000, 1000) * u.angstrom
    expected_extinction = [1.84202329, 1.42645161, 1.13844058, 0.88840962,
                           0.69220634, 0.54703201]

    model = Extinction_fm07(1.)
    calculated_extinction = model(wave).value

    np.testing.assert_array_almost_equal(expected_extinction, calculated_extinction)


@pytest.mark.parametrize(('extinction_model'), [Extinction_ccm89, Extinction_od94])
@pytest.mark.parametrize(('wavelength'), [0*u.angstrom, 1*u.m])
def test_out_of_range_simple_extinction(extinction_model, wavelength):
    with pytest.raises(ValueError):
        model = extinction_model(a_v=1.)
        x = model(wavelength)


@pytest.mark.parametrize(('extinction_model_name'), extinction_models)
def test_general_extinction_function(extinction_model_name):
    pytest.importorskip('scipy')

    wave = 5000 * u.angstrom
    a_v = 1.
    model_name = ext.__getattribute__('Extinction_{0}'.format(extinction_model_name))
    model = model_name(a_v)
    expected_extinction = model(wave)

    assert expected_extinction == extinction(wave, a_v, model=extinction_model_name)


class TestWD01():
    def setup(self):
        pytest.importorskip('scipy')
        self.extinction = Extinction_wd01(1., 3.1)

    @pytest.mark.parametrize(('wavelength'), [0*u.angstrom, 1*u.m])
    def test_out_of_range(self, wavelength):
        pytest.importorskip('scipy')
        with pytest.raises(ValueError):
            x = self.extinction(wavelength)


class TestD03():

    def setup(self):
        pytest.importorskip('scipy')
        self.extinction = Extinction_d03(1., 3.1)

    @pytest.mark.parametrize(('wavelength'), [0*u.angstrom, 1*u.m])
    def test_out_of_range(self, wavelength):
        pytest.importorskip('scipy')
        with pytest.raises(ValueError):
            x = self.extinction(wavelength)
