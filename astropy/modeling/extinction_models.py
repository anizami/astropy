# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction models.

Classes are callables representing the corresponding extinction
function with a fixed R_V.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from os import path
from ..io import ascii
from ..utils.data import get_package_data_filename
from .. import units as u
import numpy as np

from .core import (Fittable1DModel)
from .parameters import Parameter, InputParameterError
from . import cextinction

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

__all__ = ['Extinction_ccm89', 'Extinction_od94', 'Extinction_gcc09',
           'Extinction_f99', 'Extinction_fm07', 'extinction', 'Reddening',
           'Extinction_d03', 'Extinction_wd01', 'CCM']


def _process_wave(wave):
    return wave.to(u.angstrom).flatten()


def _process_inputs(wave, ebv, a_v, r_v):
    if (a_v is None) and (ebv is None):
        raise ValueError('Must specify either a_v or ebv')
    if (a_v is not None) and (ebv is not None):
        raise ValueError('Cannot specify both a_v and ebv')
    if a_v is None:
        a_v = ebv * r_v

    scalar = np.isscalar(wave)
    wave = np.atleast_1d(wave)
    if wave.ndim > 2:
        raise ValueError("wave cannot be more than 1-d")

    return wave, scalar, a_v


def _check_wave(wave, minwave, maxwave):
    if np.any((wave < minwave) | (wave > maxwave)):
        raise ValueError('Wavelengths must be between {0:.2f} and {1:.2f} '
                         'angstroms'.format(minwave, maxwave))


class Extinction_ccm89(Fittable1DModel):
    """Cardelli, Clayton, & Mathis (1989) extinction model.

    The parameters given in the original paper [1]_ are used.
    The function works between 910 A and 3.3 microns, although note the
    claimed validity is only for wavelength above 1250 A.

    The wavelength(s) should be in angstroms.

    Parameters
    ----------
    a_v : float
        A(V) total V band extinction in magnitudes
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1

    Notes
    -----
    In Cardelli, Clayton & Mathis (1989) the mean
    R_V-dependent extinction law, is parameterized as

    .. math::

       <A(\lambda)/A_V> = a(x) + b(x) / R_V

    where the coefficients a(x) and b(x) are functions of
    wavelength. At a wavelength of approximately 5494.5 angstroms (a
    characteristic wavelength for the V band), a(x) = 1 and b(x) = 0,
    so that A(5494.5 angstroms) = A_V. This function returns

    .. math::

       A(\lambda) = A_V (a(x) + b(x) / R_V)

    where A_V can either be specified directly or via E(B-V)
    (by defintion, A_V = R_V * E(B-V)).

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
    """

    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        _check_wave(x, 909.091 * u.angstrom, 33333.333 * u.angstrom)
        res = cextinction.ccm89(_process_wave(x).value, a_v, r_v)
        return res.reshape(x.shape) * u.mag

    @classmethod
    def reddening(cls, x):
        results = cls.evauluate(x, a_v, r_v)
        return 10**(0.4 * results)


class Extinction_od94(Fittable1DModel):
    """O'Donnell (1994) extinction model.

    Like Cardelli, Clayton, & Mathis (1989) [1]_ but using the O'Donnell
    (1994) [2]_ optical coefficients between 3030 A and 9091 A.

    Parameters
    ----------
    a_v : float
        A(V) total V band extinction in magnitudes
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1

    Notes
    -----
    This function matches the Goddard IDL astrolib routine CCM_UNRED.
    From the documentation for that routine:

    1. The CCM curve shows good agreement with the Savage & Mathis (1979)
       [3]_ ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    2. Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989) [4]_
    3. Valencic et al. (2004) [5]_ revise the ultraviolet CCM
       curve (3.3 -- 8.0 um^-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.

    References
    ----------
    .. [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
    .. [2] O'Donnell, J. E. 1994, ApJ, 422, 158O
    .. [3] Savage & Mathis 1979, ARA&A, 17, 73
    .. [4] Longo et al. 1989, ApJ, 339,474
    .. [5] Valencic et al. 2004, ApJ, 616, 912
    """

    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        _check_wave(x, 909.091 * u.angstrom, 33333.333 * u.angstrom)
        res = cextinction.od94(_process_wave(x).value, a_v, r_v)
        return res.reshape(x.shape) * u.mag


class Extinction_gcc09(Fittable1DModel):
    """Gordon, Cartledge, & Clayton (2009) extinction model.

    Uses the UV coefficients of Gordon, Cartledge, & Clayton (2009)
    [1]_ between 910 A and 3030 A, otherwise the same as the
    `extinction_od94` function.  Also note that the two do not connect
    perfectly: there is a discontinuity at 3030 A. Note that
    GCC09 equations 14 and 15 apply to all x>5.9 (the GCC09 paper
    mistakenly states they do not apply at x>8; K. Gordon,
    priv. comm.).

    Parameters
    ----------
    a_v : float
        A(V) total V band extinction in magnitudes
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1

    .. warning :: Note that the Gordon, Cartledge, & Clayton (2009) paper
                  has incorrect parameters for the 2175 angstrom bump that
                  have not been corrected here.

    References
    ----------
    .. [1] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320
    """

    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.m
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        _check_wave(x, 909.091 * u.angstrom, 33333.333 * u.angstrom)
        res = cextinction.gcc09(_process_wave(x).value, a_v, r_v)
        return res.reshape(x.shape) * u.mag

_f99_xknots = 1.e4 / np.array([np.inf, 26500., 12200., 6000., 5470.,
                               4670., 4110., 2700., 2600.])


class Extinction_f99(Fittable1DModel):
    """
    Fitzpatrick (1999) [1]_ model which relies on the parametrization
    of Fitzpatrick & Massa (1990) [2]_ in the UV (below 2700 A) and
    spline fitting in the optical and IR. This function is defined
    from 910 A to 6 microns, but note the claimed validity goes down
    only to 1150 A. The optical spline points are not taken from F99
    Table 4, but rather updated versions from E. Fitzpatrick (this
    matches the Goddard IDL astrolib routine FM_UNRED).

    Parameters
    ----------
    a_v : float
        A(V) total V band extinction in magnitudes
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1

    Examples
    --------
    Create a callable that gives the extinction law for a given ``r_v``
    and use it:

    >>> f = ExtinctionF99(3.1)
    >>> f(3000., a_v=1.)
    1.7993995521481463

    References
    ----------
    .. [1] Fitzpatrick, E. L. 1999, PASP, 111, 63
    .. [2] Fitzpatrick, E. L. & Massa, D. 1990, ApJS, 72, 163

    """
    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        if not HAS_SCIPY:
            raise ImportError('To use this function scipy needs to be installed')

        from scipy.interpolate import splmake, spleval

        kknots = cextinction.f99kknots(_f99_xknots, r_v)

        spline = splmake(_f99_xknots, kknots, order=3)

        wave_shape = x.shape
        wave = _process_wave(x)

        _check_wave(wave, 909.091 * u.angstrom, 6. * u.micron)

        res = np.empty_like(wave.__array__(), dtype=np.float64)

        # Analytic function in the UV.
        uvmask = wave < (2700. * u.angstrom)
        if np.any(uvmask):
            res[uvmask] = cextinction.f99uv(wave[uvmask].value, a_v, r_v)

        # Spline in the Optical/IR
        oirmask = ~uvmask
        if np.any(oirmask):
            k = spleval(spline, 1. / wave[oirmask].to('micron'))
            res[oirmask] = a_v / r_v * (k + r_v)

        return res.reshape(wave_shape) * u.mag


# fm07 knots for spline
_fm07_r_v = 3.1
_fm07_xknots = np.array([0., 0.25, 0.50, 0.75, 1., 1.e4/5530., 1.e4/4000.,
                        1.e4/3300., 1.e4/2700., 1.e4/2600.])
_fm07_kknots = cextinction.fm07kknots(_fm07_xknots)
try:
    from scipy.interpolate import splmake
    _fm07_spline = splmake(_fm07_xknots, _fm07_kknots, order=3)
except ImportError:
    pass


class Extinction_fm07(Fittable1DModel):
    """Fitzpatrick & Massa (2007) extinction model for R_V = 3.1.

    The Fitzpatrick & Massa (2007) [1]_ model, which has a slightly
    different functional form from that of Fitzpatrick (1999) [3]_
    (`extinction_f99`). Fitzpatrick & Massa (2007) claim it is
    preferable, although it is unclear if signficantly so (Gordon et
    al. 2009 [2]_). Defined from 910 A to 6 microns.

    Parameters
    ----------
    a_v : float
        A(V) total V band extinction in magnitudes

    .. note :: This model is not R_V dependent.

    References
    ----------
    .. [1] Fitzpatrick, E. L. & Massa, D. 2007, ApJ, 663, 320
    .. [2] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320
    .. [3] Fitzpatrick, E. L. 1999, PASP, 111, 63
    """
    a_v = Parameter(default=0)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v):
        if not HAS_SCIPY:
            raise ImportError('To use this model scipy needs to be installed')
        from scipy.interpolate import spleval

        wave_shape = x.shape
        wave = _process_wave(x)

        _check_wave(wave, 909.091 * u.angstrom, 6.0 * u.micron)
        res = np.empty_like(wave.__array__(), dtype=np.float64)

        # Simple analytic function in the UV
        uvmask = wave < (2700. * u.angstrom)
        if np.any(uvmask):
            res[uvmask] = cextinction.fm07uv(wave[uvmask].value, a_v)

        # Spline in the Optical/IR
        oirmask = ~uvmask
        if np.any(oirmask):
            k = spleval(_fm07_spline, (1. / wave[oirmask].to('micron')).value)
            res[oirmask] = a_v / _fm07_r_v * (k + _fm07_r_v)

        return res.reshape(wave_shape) * u.mag


prefix = path.join('data', 'extinction_models', 'kext_albedo_WD_MW')
_wd01_fnames = {'3.1': prefix + '_3.1B_60.txt',
                '4.0': prefix + '_4.0B_40.txt',
                '5.5': prefix + '_5.5B_30.txt'}
_d03_fnames = {'3.1': prefix + '_3.1A_60_D03_all.txt',
               '4.0': prefix + '_4.0A_40_D03_all.txt',
               '5.5': prefix + '_5.5A_30_D03_all.txt'}
del prefix


class Extinction_wd01(Fittable1DModel):
    """Weingartner and Draine (2001) extinction model.

    The Weingartner & Draine (2001) [1]_ dust model.  This model is a
    calculation of the interstellar extinction using a dust model of
    carbonaceous grains and amorphous silicate grains. The
    carbonaceous grains are like PAHs when small and like graphite
    when large. This model is evaluated at discrete wavelengths and
    interpolated between these wavelengths. Grid goes from 1 A to 1000
    microns. The model has been calculated for three different grain
    size distributions which produce interstellar exinctions that look
    like 'ccm89' at Rv = 3.1, Rv = 4.0 and Rv = 5.5.  No interpolation
    to other Rv values is performed, so this model can be evaluated
    only for these values.

    The dust model gives the extinction per H nucleon.  For
    consistency with other extinction laws we normalize this
    extinction law so that it is equal to 1.0 at 5495 angstroms.

    .. note :: Model is not an analytic  function of R_V. Only ``r_v``
               values of 3.1, 4.0 and 5.5 are accepted.


    Parameters
    ----------
    r_v : float
        Relation between specific and total extinction, ``a_v = r_v * ebv``.

    Examples
    --------
    Create a callable that gives the extinction law for a given ``r_v``
    and use it:

    >>> f = Extinction_wd01(3.1)
    >>> f(3000., a_v=1.)

    Arrays are also accepted and ``ebv`` can be specified instead of ``a_v``:

    >>> f([3000., 4000.], ebv=1./3.1)

    References
    ----------
    .. [1] Weingartner, J.C. & Draine, B.T. 2001, ApJ, 548, 296


    """

    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        # TODO: Need to refactor this and make spline some instance property
        if not HAS_SCIPY:
            raise ImportError('To use this function scipy needs to be installed')

        from scipy.interpolate import interp1d

        fname_key = [item for item in _wd01_fnames.keys() if np.isclose(
            float(item), r_v)]

        if len(fname_key) == 0:
            raise ValueError("model only defined for r_v in [3.1, 4.0, 5.5]")
        elif len(fname_key) == 1:
            fname = _wd01_fnames[fname_key[0]]
        else:
            raise ValueError('The given float {0} matches multiple available'
                             ' r_vs [3.1, 4.0, 5.5] - unexpected code error')

        fname = get_pkg_data_filename(fname)
        data = ascii.read(fname, Reader=ascii.FixedWidth, data_start=51,
                          names=['wave', 'albedo', 'avg_cos', 'C_ext',
                                 'K_abs'],
                          col_starts=[0, 10, 18, 25, 35],
                          col_ends=[9, 17, 24, 34, 42], guess=False)

        # Reverse entries so that they ascend in x (needed for the spline).
        waveknots = np.asarray(data['wave'])[::-1]
        cknots = np.asarray(data['C_ext'])[::-1]
        xknots = 1. / waveknots  # Values in inverse microns.

        # Create a spline just to get normalization.
        spline = interp1d(xknots, cknots)
        cknots = cknots / spline(1.e4 / 5495.)  # Normalize cknots.
        spline = interp1d(xknots, cknots)

        wave_shape = x.shape
        wave = _process_wave(x)

        x = (1 / wave).to('1/micron')

        res = a_v * spline(x.value)

        return res.reshape(wave_shape) * u.mag


class Extinction_d03(Fittable1DModel):
    """Draine (2003) extinction model.

    The Draine (2003) [2]_ update to wd01 [1]_ where the
    carbon/PAH abundances relative to 'wd01' have been reduced by a
    factor of 0.93.

    The dust model gives the extinction per H nucleon.  For
    consistency with other extinction laws we normalize this
    extinction law so that it is equal to 1.0 at 5495 angstroms.

    Parameters
    ----------
    r_v : float
        Relation between specific and total extinction, ``a_v = r_v * ebv``.

    .. note :: Model is not an analytic  function of R_V. Only ``r_v``
               values of 3.1, 4.0 and  5.5 are accepted.

    Examples
    --------
    Create a callable that gives the extinction law for a given ``r_v``
    and use it:

    >>> f = Extinction_wd01(3.1)
    >>> f(3000., a_v=1.)

    Arrays are also accepted and ``ebv`` can be specified instead of ``a_v``:

    >>> f([3000., 4000.], ebv=1./3.1)

    References
    ----------
    .. [1] Weingartner, J.C. & Draine, B.T. 2001, ApJ, 548, 296
    .. [2] Draine, B.T. 2003, ARA&A, 41, 241

    """

    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    input_units = u.angstrom
    output_units = u.mag

    @staticmethod
    def evaluate(x, a_v, r_v):
        # TODO: Need to refactor this and make spline some instance property
        if not HAS_SCIPY:
            raise ImportError('To use this function scipy needs to be installed')

        from scipy.interpolate import interp1d

        fname_key = [item for item in _wd01_fnames.keys() if np.isclose(
            float(item), r_v)]

        if len(fname_key) == 0:
            raise ValueError("model only defined for r_v in [3.1, 4.0, 5.5]")
        elif len(fname_key) == 1:
            fname = _d03_fnames[fname_key[0]]
        else:
            raise ValueError('The given float {0} matches multiple available'
                             ' r_vs [3.1, 4.0, 5.5] - unexpected code error')

        fname = get_pkg_data_filename(fname)

        data = ascii.read(fname, Reader=ascii.FixedWidth, data_start=67,
                          names=['wave', 'albedo', 'avg_cos', 'C_ext',
                                 'K_abs', 'avg_cos_sq', 'comment'],
                          col_starts=[0, 12, 20, 27, 37, 47, 55],
                          col_ends=[11, 19, 26, 36, 46, 54, 80], guess=False)
        xknots = 1. / np.asarray(data['wave'])
        cknots = np.asarray(data['C_ext'])

        # Create a spline just to get normalization.
        spline = interp1d(xknots, cknots)
        cknots = cknots / spline((1. / (5495. * u.angstrom)).to('1/micron').value)  # Normalize cknots.
        spline = interp1d(xknots, cknots)

        wave_shape = x.shape
        wave = _process_wave(x)

        x = (1 / wave).to('1/micron')

        res = a_v * spline(x.value)

        return res.reshape(wave_shape) * u.mag


_extinction_models = {'ccm89': Extinction_ccm89,
                      'od94': Extinction_od94,
                      'gcc09': Extinction_gcc09,
                      'f99': Extinction_f99,
                      'fm07': Extinction_fm07,
                      'wd01': Extinction_wd01,
                      'd03': Extinction_d03}


def extinction(wave, a_v, r_v=3.1, model='od94'):
    """Generic interface for all extinction model functions.

    Parameters
    ----------
    wave : float or list_like
        Wavelength(s) in angstroms.
    a_v : float
        Total V band extinction A(V), in magnitudes. A(V) = R_V * E(B-V).
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1.
    model : {'ccm89', 'od94', 'gcc09', 'f99', 'fm07', 'wd01', d03'}, optional
        Use function ``extinction_[model]``. E.g., for 'ccm89', the function
        ``Extinction_ccm89`` is used.

    Returns
    -------
    extinction : float or `~numpy.ndarray`
        Extinction in magnitudes at given wavelengths.

    See Also
    --------
    Extinction_ccm89
    Extinction_od94
    Extinction_gcc09
    Extinction_f99
    Extinction_fm07
    Extinction_wd01
    Extinction_d03
    Reddening

    Notes
    -----

    **Visual comparison of models**

    The plot below shows a comparison of the models for
    ``r_v=3.1``. The shaded regions show the limits of claimed
    validity for the f99 model (> 1150 A) and the ccm89/od94 models (>
    1250 A). The vertical dotted lines indicate transition wavelengths in
    the models (3030.3 A and 9090.9 A for ccm89, od94 and gcc09;
    2700. A for f99, fm07).

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from mpl_toolkits.axes_grid1 import make_axes_locatable
       from specutils.extinction import extinction

       models = ['ccm89', 'od94', 'gcc09', 'f99', 'fm07','wd01','d03']
       wave = np.logspace(np.log10(910.), np.log10(30000.), 2000)
       a_lambda = {model: extinction(wave, a_v=1., model=model)
                   for model in models}

       fig = plt.figure(figsize=(8.5, 6.))

       ax = plt.axes()
       for model in models:
           plt.plot(wave, a_lambda[model], label=model)
       plt.axvline(x=2700., ls=':', c='k')
       plt.axvline(x=3030.3030, ls=':', c='k')
       plt.axvline(x=9090.9091, ls=':', c='k')
       plt.axvspan(wave[0], 1150., fc='0.8', ec='none', zorder=-1000)
       plt.axvspan(1150., 1250., fc='0.9', ec='none', zorder=-1000)
       plt.text(0.67, 0.95, '$R_V = 3.1$', transform=ax.transAxes, va='top',
                size='x-large')
       plt.ylabel('Extinction ($A(\lambda)$ / $A_V$)')
       plt.legend()
       plt.setp(ax.get_xticklabels(), visible=False)

       divider = make_axes_locatable(ax)
       axresid = divider.append_axes("bottom", size=2.0, pad=0.2, sharex=ax)
       for model in models:
           plt.plot(wave, a_lambda[model] - a_lambda['f99'])
       plt.axvline(x=2700., ls=':', c='k')
       plt.axvline(x=3030.3030, ls=':', c='k')
       plt.axvline(x=9090.9091, ls=':', c='k')
       plt.axvspan(wave[0], 1150., fc='0.8', ec='none', zorder=-1000)
       plt.axvspan(1150., 1250., fc='0.9', ec='none', zorder=-1000)
       plt.xlim(wave[0], wave[-1])
       plt.ylim(ymin=-0.4,ymax=0.4)
       plt.ylabel('residual from f99')
       plt.xlabel('Wavelength ($\AA$)')

       ax.set_xscale('log')
       axresid.set_xscale('log')
       plt.tight_layout()

       fig.show()

    Examples
    --------

    >>> wave = [2000., 2500., 3000.]
    >>> extinction(wave, a_v=1., r_v=3.1, model='f99')
    array([ 2.76225609,  2.27590036,  1.79939955])

    The extinction scales linearly with ``a_v``. This means
    that when calculating extinction for multiple values of ``a_v``, one can compute extinction ahead of time for a given set of
    wavelengths and then scale by ``a_v`` or ``ebv`` later.  For example:

    >>> a_lambda_over_a_v = extinction(wave, a_v=1.)
    >>> a_v = 0.5
    >>> a_lambda = a_v * a_lambda_over_a_v

    """

    model = model.lower()
    if model not in _extinction_models:
        raise ValueError('unknown model: {0}'.format(model))

    if model == 'fm07':
        if not np.isclose(r_v, 3.1):
            raise ValueError('r_v must be 3.1 for fm07 model')
        model = _extinction_models[model](a_v=a_v)
    else:
        model = _extinction_models[model](a_v=a_v, r_v=r_v)
    return model(wave)


class Reddening(Fittable1DModel):
    """Inverse of flux transmission fraction at given wavelength(s).

    Parameters
    ----------
    wave : float or list_like
        Wavelength(s) in angstroms at which to evaluate the reddening.
    a_v : float
        Total V band extinction, in magnitudes. A(V) = R_V * E(B-V).
    r_v : float, optional
        R_V parameter. Default is the standard Milky Way average of 3.1.
    model : {'ccm89', 'od94', 'gcc09', 'f99', 'fm07'}, optional

    Returns
    -------
    reddening : float or `~numpy.ndarray`
        Inverse of flux transmission fraction, equivalent to
        ``10**(0.4 * extinction(wave))``. To deredden spectra,
        multiply flux values by these value(s). To redden spectra, divide
        flux values by these value(s).

    """
    a_v = Parameter(default=0)
    r_v = Parameter(default=3.1)

    @staticmethod
    def evaluate(x, a_v, r_v, model='od94'):
        result = extinction(x, a_v, r_v=r_v, model=model).value
        return 10**(0.4 * result)


class CCM(Fittable1DModel):
    '''
    Computes reddening correction according to the Cardelli, Clayton and Mathis
    model (ApJ 1989 v345, p245)

    Parameters
    ----------
    wave : float or numpy array
        Wavelength(s) in microns
    r_v : float, optional
        R_V parameter. Rv = A(V)/E(B-V).
    ebmv : float, optional
            e(B-V) in magnitudes
    '''
    r_v = Parameter(default=3.5)
    ebmv = Parameter(default=1.)

    @staticmethod
    def evaluate(x, ebmv, r_v):
        x = 1./x
        irmask = (x >= 0.3) & (x <= 1.1)
        omask = (x > 1.1) & (x <= 3.3)
        nuvmask1 = (x > 3.3) & (x <= 8.0)
        nuvmask2 = (x >= 5.9) & (x <= 8.0)
        fuvmask = (x > 8.0) & (x <= 20.)
        a = 0*x
        b = 0*x
        # IR
        xir = x[irmask]**1.61
        a[irmask] = 0.574 * xir
        b[irmask] = -0.527 * xir
        # Optical (could do this better numerically)
        xopt = x[omask] - 1.82
        a[omask] = (1.0 + 0.17699 * xopt - 0.50477 * xopt**2 - 0.02427 * xopt**3 +
                    0.72085 * xopt**4 + 0.01979 * xopt**5 - 0.77530 * xopt**6 +
                    0.32999 * xopt**7)
        b[omask] = (0.0 + 1.41338 * xopt + 2.28305 * xopt**2 + 1.07233 * xopt**3 -
                    5.38434 * xopt**4 - 0.62551 * xopt**5 + 5.30260 * xopt**6 -
                    2.09002 * xopt**7)
        # Near UV
        xnuv1 = x[nuvmask1]
        a[nuvmask1] = 1.752 - 0.316 * xnuv1 - 0.104 / (0.341 + (xnuv1-4.67)**2)
        b[nuvmask1] = - 3.090 + 1.825 * xnuv1 + 1.206 / (0.263 + (xnuv1-4.62)**2)
        xnuv2 = x[nuvmask2] - 5.9
        a[nuvmask2] += -0.04473 * xnuv2**2 - 0.009779 * xnuv2**3
        b[nuvmask2] += 0.21300 * xnuv2**2 + 0.120700 * xnuv2**3

        # Far UV
        xfuv = x[fuvmask] - 8.0
        a[fuvmask] = -1.073 - 0.628 * xfuv + 0.137 * xfuv**2 - 0.070 * xfuv**3
        b[fuvmask] = 13.670 + 4.257 * xfuv - 0.420 * xfuv**2 + 0.374 * xfuv**3

        return 10 ** (-0.4 * ebmv * (r_v * a + b))
