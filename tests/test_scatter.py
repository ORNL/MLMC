from mlmc.xp import xp
from mlmc.scatter.scatter import Scatter

import typer
app = typer.Typer()

def test_scatter():
    L = xp.array([[5.0, 0.0], [0.0, 5.0]])
    S = Scatter(L, 3.0)

    r = xp.array( [[2.5, 2.5],[1.1, 0.1],[5.0-1.1, -0.1]] ) @ L
    E = S.E(r)
    assert E.shape == (3,)
    dE = S.dE(r)
    assert dE.shape == (3,2)

    assert xp.abs(E[0]) < 1e-8
    assert xp.abs(dE[0]).max() < 1e-8

    # points 1 and 2 are related by reflection
    # and translation by [5,0]
    assert xp.abs(E[1]-E[2]) < 1e-8
    assert xp.abs(dE[1]+dE[2]).max() < 1e-8

def test_du_scatter():
    dr = 0.0001
    r = xp.array([[1.1, 0.0], [1.1-dr, 0.0], [1.1+dr, 0.0]])

    L = xp.array([[3.3, 0.0], [0.2, 3.1]])
    S = Scatter(L, 3.0)
    
    u = S.E(r)
    assert u.shape == (3,)

    du = S.dE(r)
    assert du.shape == (3,2)
    num = (u[2]-u[1]) / (2*dr)
    print(num)
    print(du[0])
    assert abs(num - du[0,0]) < dr
    assert abs(du[0,1]) < 1e-8

@app.command()
def run():
    print('Testing scatter...')
    test_scatter()

    print()
    print('Testing scatter derivative...')
    test_du_scatter()

if __name__ == '__main__':
    run()
