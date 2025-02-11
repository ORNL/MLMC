from mlmc.xp import xp
from mlmc.scatter.scatter import LJ_cut

def test_u_lj():
    r0 = xp.array([0.0, 1.0])
    r1 = xp.array([[1.0, 0.0], [1.2, 0.0]])
    r2 = xp.array([
                [[0.0, 1.0], [0.0, 1.2], [0.1, 1.2]],
                [[0.2, 0.8], [0.3, 0.8], [0.3, 0.9]]
                ])

    lj = LJ_cut(3.3)
    
    u0 = lj.E(r0)
    assert isinstance(u0, float) or u0.shape == ()

    u1 = lj.E(r1)
    assert u1.shape == (2,)

    u2 = lj.E(r2)
    assert u2.shape == (2,3)

def test_du_lj():
    dr = 0.0001
    r = xp.array([[1.1, 0.0], [1.1-dr, 0.0], [1.1+dr, 0.0]])

    lj = LJ_cut(3.3)
    
    u = lj.E(r)
    assert u.shape == (3,)

    du = lj.dE(r)
    assert du.shape == (3,2)
    num = (u[2]-u[1]) / (2*dr)
    print(num)
    print(du[0])
    assert abs(num - du[0,0]) < dr
    assert abs(du[0,1]) < 1e-8

def test_d2u_lj():
    dr = 0.0001
    r = xp.empty((5,2))
    r[:] = xp.array([1.1, 0.1])[None,:]
    r[1,0] -= dr
    r[2,0] += dr
    r[3,1] -= dr
    r[4,1] += dr

    lj = LJ_cut(3.3)
    
    du = lj.dE(r)
    assert du.shape == (5,2)

    d2u = lj.d2E(r)
    assert d2u.shape == (5,2,2)

    num = xp.empty((2,2))
    num[0] = (du[2]-du[1])/(2*dr)
    num[1] = (du[4]-du[3])/(2*dr)
    print(num)
    print(d2u[0])
    assert abs(num - d2u[0]).max() < dr

if __name__ == '__main__':
    print('Testing Lennard-Jones potential...')
    test_u_lj()

    print()
    print('Testing LJ derivative...')
    test_du_lj()

    print()
    print('Testing LJ second derivative...')
    test_d2u_lj()