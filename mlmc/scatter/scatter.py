from mlmc.xp import xp

def u_lj(r2):
    # compute the LJ potential for a given r2 = r*r
    #
    # Note, an equivalent representation is:
    #   r**-12 - 2*r**-6 = 4*(x**-12 - x**-6)
    #   r = a*x, a**-6 = 2
    #
    # r2 can be any shape
    #
    ir2 = 1.0/r2
    ir6 = ir2*ir2*ir2
    return (ir6 - 2.0)*ir6

def du_lj_fac(r2):
    # compute the prefactor used in du_lj below
    ir2 = 1.0/r2
    ir6 = ir2*ir2*ir2
    ir8 = ir6*ir2
    return 12.0 * (1.0 - ir6)*ir8

def du_lj(r):
    # compute d(u_lj)/dr_0, d(u_lj)/dr_1]
    # = r/|r| * -12*(r**-13 - r**-7)
    # = -12*r * (r**-14 - r**-8)
    #
    # r can be (any shape,) + (3,)
    # and the result will come out in the
    # same shape as r.
    #
    r2 = (r*r).sum(-1) # sum over last axis
    return r * du_lj_fac(r2)[..., None]

def d2u_lj_fac(r2):
    # compute both prefactors used in d2u_lj below
    ir2  = 1.0/r2
    ir6  = ir2*ir2*ir2
    ir8  = ir6*ir2
    ir10 = ir8*ir2
    return 12.0 * (1.0 - ir6)*ir8, \
            ((168.0)*ir6 - (96.0))*ir10

def d2u_lj(r):
    # compute d2(u_lj)/dr^2
    # = -12*( I *(  r^-14 -    r^-8)
    #       + rr*(8*r^-10 - 14*r^-16)
    #
    r2 = (r*r).sum(-1)
    a, b = du_lj_fac(r2)
    rr = r[...,:,None]*r[...,None,:]
    return a[...,None,None]*xp.eye(2) + b[...,None,None]*rr

class LJ_cut:
    """ An LJ potential with a fixed distance cutoff,

        U = { U_LJ(r) - U_LJ(Rc), r <= Rc
              0, otherwise
            }

        Note this means that dU/dr has a jump
        disconuity at Rc unless Rc = 1 (the WCA potential).
    """
    def __init__(self, Rc):
        self.Rc2 = Rc**2
        self.shift = u_lj(self.Rc2)
    def E(self, r):
        r2 = (r*r).sum(-1)
        return (u_lj(r2) - self.shift) * (r2 <= self.Rc2)
    def dE(self, r):
        r2 = (r*r).sum(-1)
        fac = du_lj_fac(r2)
        fac *= r2 <= self.Rc2
        return r * fac[..., None]
    def d2E(self, r):
        r2 = (r*r).sum(-1)
        a, b = d2u_lj_fac(r2)
        a *= r2 <= self.Rc2
        b *= r2 <= self.Rc2
        rr = r[...,:,None]*r[...,None,:]
        return a[...,None,None]*xp.eye(2) + b[...,None,None]*rr

class Scatter:
    """ Python class representing the potential generated
        by a fixed set of scatterers.

        This class mainly exists to deal with the
        lattice sum defined for periodically replicated

        L is an array with L[0] = lattice vector 0
        and L[1] = lattice vector 1.

        The lattice must be rotated so that L[0]
        lies along the x-axis and L[1,0] has been
        "wrapped" along x to the shortest possible
        value (L[1,0] - L[0,0]*floor(L[1,0]/L[0,0]+0.5)).

        Note that only the 2x2 lattice points nearest
        the box [0,0]:a:b are summed into the
        potential.
        This assumes that |a| = L[0,0] > Rc and L[1,1] > Rc.
        Larger cutoffs can be handled, but require
        more lattice points.
    """
    def __init__(self, L, Rc):
        self.L = L
        assert L[0,1] == 0.0, "Axis 0 should be along x"
        assert abs(L[1,0]) <= 0.5*L[0,0]+1e-8, "Axis 1 should be minimal."
        assert L[0,0] >= Rc, "Short a-vector not supported."
        assert L[1,1] >= Rc, "Short b-vector not supported."
        # Lattice vectors within distance Rc from
        # a rectangle with lengths L[0,0], L[1,1]
        # centered at the origin
        self.vecs = xp.array([[0, 0], [1, 0],
                              [0, 1], [1, 1]]) @ L

        self.LJ = LJ_cut(Rc)

    def wrap(self, r):
        """ Wrap r into the box [0,0]:a:b
        """
        # L^-1 = [[L[1,1], 0], [-L[1,0], L[0,0]] / A
        L = self.L
        A = L[0,0]*L[1,1]
        kx = (1.0/L[0,0])*r[...,0] - (L[1,0]/A)*r[...,1]
        ky = (1.0/L[1,1])*r[...,1]
        kx -= xp.floor(kx)
        ky -= xp.floor(ky)

        return kx[...,None]*L[0] + ky[...,None]*L[1]

    def E(self, r):
        rr = self.wrap(r)
        dr = rr[...,None,:] - self.vecs
        u = self.LJ.E(dr)
        return u.sum(-1)

    def dE(self, r):
        rr = self.wrap(r)
        dr = rr[...,None,:] - self.vecs
        du = self.LJ.dE(dr)
        return du.sum(-2)

    def d2E(self, r):
        rr = self.wrap(r)
        dr = rr[...,None,:] - self.vecs
        d2u = self.LJ.d2E(dr)
        return d2u.sum(-3)
