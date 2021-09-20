import numpy as np
from scipy.special import erf

# TODO: incorporate angular dependence

hbar = 6.582e-16  # eV s
c = 3e10  # cm / s

# https://jameslandry.files.wordpress.com/2012/02/09-appendix-b.pdf


def kk_transform(kk_inv):
    """kramers kronig via discrete hilbert transform
    https://en.wikipedia.org/wiki/Hilbert_transform#Discrete_Hilbert_transform
    """
    n = np.arange(kk_inv.size) - kk_inv.size // 2
    n[::2] = np.inf
    kernel = -2 / (np.pi * n)
    # kernel[::2] = 0
    
    kk = np.convolve(kk_inv, kernel, mode='same')
    return kk


def n_air(w):
    if isinstance(w, float) or isinstance(w, int):
        return 1
    return np.ones(w.shape, dtype=complex)


def n_fused_silica(w):
    if isinstance(w, float) or isinstance(w, int):
        return 1.46
    return np.ones(w.shape, dtype=complex) * 1.46


def generate_n_qw(E_gap, E_b, alpha0=1e5, eh_x_ratio=0.5, width=0.03, baseline=1):
    """generate refractive index that mimics a 2D QW band edge
    note: function needs a large range of values to get the kk transform correct
    """
    # todo: consistent units
    # todo: define args

    def n_qw(E):
        # work through absorptivity
        continuum = erf((E - E_gap) / (2**0.5 * width)) + 1
        continuum *= alpha0 * eh_x_ratio / 2
        z = (E - E_gap + E_b) / (2**0.5 * width)
        exciton = alpha0 * np.exp(-z**2)
        
        alpha = continuum + exciton
        w = E / hbar
        chi_im = alpha * c * baseline / w
        chi_re = kk_transform(chi_im)

        n = np.sqrt(baseline**2 + chi_re + 1j * chi_im)

        return n

    return n_qw


def Mi(ri, ti, deltai):
    m_propagate = np.zeros(ri.shape + (2, 2), dtype=complex)
    m_propagate[..., 0, 0] = np.exp(-1j * deltai)
    m_propagate[..., 1, 1] = np.exp(1j * deltai)
    m_interface = np.ones(m_propagate.shape, dtype=complex)
    m_interface[..., 0, 1] = ri
    m_interface[..., 1, 0] = ri
    m_interface /= ti[..., None, None]
    return m_propagate @ m_interface


def gen_r_layer(n1, n2, angle1=None, angle2=None):
    if angle1 is None:
        angle1 = lambda E: E*0
    if angle2 is None:
        angle2 = lambda E: E*0
    def r_layer_spol(E):
        # s-polarization
        x1 = n1(E) * np.cos(angle1(E))
        x2 = n2(E) * np.cos(angle2(E))
        return (x1 - x2) / (x1 + x2)
    return r_layer_spol


def gen_t_layer(n1, n2, angle1=None, angle2=None):
    if angle1 is None:
        angle1 = lambda E: E*0
    if angle2 is None:
        angle2 = lambda E: E*0
    def t_layer_spol(E):
        # s-polarization
        x1 = n1(E) * np.cos(angle1(E))
        x2 = n2(E) * np.cos(angle2(E))
        return 2 * x1 / (x1 + x2)
    return t_layer_spol


def gen_delta(ni, di, angle=None):
    if angle is None:
        angle1 = lambda E: E*0       
    def delta(E):
        w = E / hbar
        c = 3e10  # cm / s
        return di * ni(E) * np.cos(angle(E)) * w / c
    return delta


def gen_angle(n1, n2, incident_angle):
    """
    calc angles between layers based on snells law
    currently assumes refraction is negligible
    todo: handle internal reflection
    """
    def angle(E):
        return np.arcsin(n1(E) / n2(E) * np.sin(incident_angle(E)))
    return angle

class FilmStack:
    def __init__(self, ns, ds, angle=0):
        """define a film of stacked materials
        parameters
        ----------
        ns : list of functions describing the optical constant, n, as a function of energy (eV).
            Elements are ordered from front to back.
        ds : list of length `len(ns)-2`
            list of thicknesses (units of centimeters) for the interior materials of the stack
        angle : angle of incidence for light on front surface, in radians
            default is normal incidence (0)
        """
        self.ns = ns
        self.angles = [lambda E: angle]
        for i in range(len(self.ns)-1):
            self.angles.append(gen_angle(ns[i], ns[i+1], self.angles[-1]))
        self.rs = [gen_r_layer(ns[i], ns[i+1], self.angles[i], self.angles[i+1]) for i in range(len(ns)-1)]
        self.ts = [gen_t_layer(ns[i], ns[i+1]) for i in range(len(ns)-1)]
        self.ds = ds
        self.deltas = [gen_delta(ns[i+1], ds[i], self.angles[i+1]) for i in range(len(ds))]


    def M(self, E):
        """generates fresnel transfer matrix for photon energies E
        generalized to handle arbitrary layer numbers
        """
        M = None
        if len(self.rs) > 1:
            for i in range(len(self.rs)-1, 0, -1):
                ri = self.rs[i](E)
                ti = self.ts[i](E)
                deltai = self.deltas[i-1](E)
                mi = Mi(ri, ti, deltai)
                if M is None:
                    M = mi
                else:
                    M = mi @ M
        else:
            M = np.eye(2)[None, :, :]
        r0 = self.rs[0](E)
        t0 = self.ts[0](E)
        m0 = np.ones(r0.shape + (2,2), dtype=complex)
        m0[..., 0,1] = r0
        m0[..., 1,0] = r0
        m0 = m0 / t0[..., None, None]
        M = m0 @ M
        return M

    def rt(self, E):
        """Calculate reflectivity and transmissivity coefficients
        """
        M = self.M(E)
        t_frontside = 1 / M[..., 0, 0]
        r_frontside = t_frontside * M[..., 1, 0]
        t_backside = M[..., 1, 1] - M[..., 1, 0] * M[..., 0, 1] / M[..., 0, 0]
        r_backside = - t_frontside * M[..., 0, 1]
        return r_frontside, t_frontside, r_backside, t_backside

    def RT(self, E):
        """Reflectance and Transmittance
        """
        rf, tf, rb, tb = self.rt(E)
        return (
            np.abs(rf)**2,
            np.abs(tf)**2 * self.ns[-1](E).real / self.ns[0](E).real,
            np.abs(rb)**2,
            np.abs(tb)**2 * self.ns[0](E).real / self.ns[-1](E).real
        )    

    def _r(self, E):
        """computes reflectivity coefficient.  only for 4 layer sample
        """
        rs = self.rs
        deltas = self.deltas

        c0 = np.exp(1j * 2 * deltas[0](E))
        c1 = np.exp(1j * 2 * deltas[1](E))
        out = rs[0](E) + rs[1](E) * c0 \
            + rs[2](E) * c0 * c1 \
            + rs[0](E) * rs[1](E) * rs[2](E) * c1
        out /= 1 \
            + rs[0](E) * rs[1](E) * c0 \
            + rs[0](E) * rs[2](E) * c0 * c1 \
            + rs[1](E) * rs[2](E) * c1
        return out

    def _t(self, E):
        """computes stack transmission coefficient.  only for 4 layer sample
        """
        rs = self.rs
        ts = self.ts
        deltas = self.deltas

        c0 = np.exp(1j * deltas[0](E))
        c1 = np.exp(1j * deltas[1](E))

        out = ts[0](E) * ts[1](E) * ts[2](E) * c0 * c1
        out /= 1 \
            + rs[0](E) * rs[1](E) * c0**2 \
            + rs[1](E) * rs[2](E) * c1**2 \
            + rs[0](E) * rs[2](E) * c1**2 * c0**2
        return out

