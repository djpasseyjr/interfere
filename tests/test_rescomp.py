from interfere._methods.reservoir_computer import ResComp
import numpy as np
import itertools
import scipy as sp

PARAMCOMBOS  =  {
    "res_sz" : [100, 500],
    "activ_f" : [np.tanh, np.sin],
    "mean_degree" : [2.0, 3.0],
    "ridge_alpha" : [1e-4, 1.0],
    "spect_rad" : [.9, 2.0],
    "sparse_res" : [True, False],
    "sigma" : [0.1, 0.5],
    "uniform_weights" : [True, False],
    "gamma" : [1., 5.],
    "max_weight" : [10, 1],
    "min_weight" : [0, -1]
}

RES = {
    "res_sz": 50,
    "activ_f": np.tanh,
    "mean_degree": 2.0,
    "ridge_alpha": .0001,
    "spect_rad": 0.5,
    "gamma": 5.,
    "sigma": 1.5,
    "uniform_weights": True,
    "sparse_res": True,
    "map_initial" : "activ_f"
}

ADJCOMBOS = {
    "res_sz" : [100, 500],
    "rho" : [1.5, 2.0],
    "sparse" : [True, False]
}

def params_match(rcomp, keys, prms):
    for k, p in zip(keys, prms):
        rc_val = rcomp.__dict__[k]
        if  rc_val != p:
            if k in ["min_weight", "max_weight"]:
                # Reservoir edge weights are scaled to achieve
                # the correct spectral radius
                pass
            elif type(p) is not float:
                print("\n", k, rc_val, p, 1)
                return False
            elif np.abs(rc_val - p) > 0.1:
                print("\n", k, rc_val, p, 2)
                return False
    return True


def fixed_point_err(rcomp):
    t, U = make_data()
    rcomp.train(t, U)
    u0 = np.random.rand(rcomp.signal_dim_)
    u = lambda x: u0
    fixed_res_ode = lambda r: rcomp.res_ode(0, r, u)
    rstar_iter = sp.optimize.fsolve(fixed_res_ode, np.ones(rcomp.res_sz))
    rcomp.map_initial = "relax"
    rstar_relax = rcomp.initial_condition(u0)
    return np.max(np.abs(rstar_relax - rstar_iter))

def identity_adj(n=150, rho=1.0, sparse=False):
    A = sp.sparse.eye(n)
    if not sparse:
        A = A.toarray()
    A = rho * A
    datamembers = (n, np.tanh, 1.0, 1e-4, rho, sparse, 0.1, True, 1.0, 3, rho, rho)
    return A, datamembers

def nonuniform_adj(n=100, rho=1.0, sparse=False):
    A = sp.sparse.random(n, n, format='lil')
    A[0,0] = 1 # To ensure non-zero spectral radius
    spect_rad = np.max(np.abs(np.linalg.eigvals(A.toarray())))
    A = A / spect_rad
    if not sparse:
        A = A.toarray()
    A = rho * A
    maxw = float(rho / spect_rad)
    args = (n, np.tanh, n*0.01, 1e-4, rho, sparse, 0.1, False, 1.0, 3, maxw, 0.1)
    return A, args

def make_data():
    t = np.linspace(0, 20, 1000)
    U =  np.vstack((np.cos(t), -1 * np.sin(t))).T
    return t, U

def make_train_test_data():
    tr = np.linspace(0,20, 1000)
    ts = np.linspace(20,25, 500)
    signal = lambda x: np.vstack((np.cos(x), -1 * np.sin(x))).T
    return tr, ts, signal(tr), signal(ts)

# Test partition
def random_time_array(n, start=0):
    t = [start]
    def nextsample(t):
        t[0] += np.random.rand()
        return t[0]
    return [nextsample(t) for i in range(n)]

def uniform_time_array(n, start=0, end=500):
    return np.linspace(start, end, n)


def test_init_noargs():
    kwargs = dict()
    combos = itertools.product(*PARAMCOMBOS.values())
    keys = list(PARAMCOMBOS.keys())
    for c in combos:
        # For each combination of parameters, make a dictionary of kwargs
        for k, v in zip(keys, c):
            kwargs[k] = v
        # Initialize a reservoir computer
        rcomp = ResComp(**kwargs)
        # Check that the initialized rcomp has the right internal data
        assert params_match(rcomp, keys, c)


def test_drive():
    """ Drive the internal ode """
    t, U = make_data()
    rcomp = ResComp(**RES)
    rcomp.train(t, U)
    r0 = rcomp.W_in_ @ U[0, :]
    out = rcomp.internal_state_response(t, U, np.zeros((len(t), 1)), r0)
    m, n = out.shape
    assert m == len(t) and n == rcomp.res_sz

def test_fit():
    """ Make sure updates occur in the Tikhanov Factors"""
    rcomp = ResComp(**RES)
    t, U = make_data()
    rcomp.train(t, np.zeros_like(U))

    rcomp.update_tikhanov_factors(t, U, np.zeros((len(t), 1)))
    assert not np.all(rcomp.Rhat_ == 0.0)
    assert not np.all(rcomp.Yhat_ == 0.0)

def test_forecast():
    """ Test that the reservoir can learn a simple signal"""
    rcomp = ResComp(**RES)
    t, U = make_data()
    rcomp.train(t, U)
    pre = rcomp.forecast(t[500:], u0=U[500, :])
    error = np.max(np.linalg.norm(pre - U[500:, :], ord=np.inf, axis=0))
    assert error < 0.5

def test_predict_unseen():
    """ Predict on unseen data """
    rcomp = ResComp(**RES, window=10, overlap=0.9)
    tr, ts, Utr, Uts = make_train_test_data()
    rcomp.train(tr, Utr)
    pre = rcomp.forecast(ts, u0=Uts[0, :])
    error = np.mean(np.linalg.norm(pre - Uts, ord=2, axis=0)**2)**(1/2)
    assert error < 1.0

def test_window():
    """ Make sure each partition is smaller than the given time window """
    rcomp = ResComp(**RES)
    for window in [.5, 3, 1001]:
        for timef in [random_time_array, uniform_time_array]:
            times = timef(1000)
            idxs = rcomp._partition(times, window, 0)
            for i,j in idxs:
                sub = times[i:j]
                assert sub[-1] - sub[0] <= window + 1e-12

def test_overlap():
    """ Ensure that overlap is correct on average """
    rcomp = ResComp(**RES)
    for window in [30, 100]:
        for overlap in [.1, .9,]:
            T = 1000
            for times in [random_time_array(T), uniform_time_array(T)]:
                idxs = rcomp._partition(times, window, overlap)
                prev = None
                over = 0.0
                for i,j in idxs:
                    sub = times[i:j]
                    if prev is not None:
                        inters = set(sub).intersection(set(prev))
                        over += len(inters) / len(sub)
                    prev = sub
                assert np.abs(over/len(idxs) - overlap) < .05
