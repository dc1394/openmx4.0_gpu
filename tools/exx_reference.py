#!/usr/bin/env python3
"""
exx_reference.py -- reference implementation ("oracle") of the exact-exchange
matrix build described in EXX_IMPLEMENTATION_GUIDE.md (OpenMX 4.0 GPU, HSE06/PBE0).

Contents
  Model               periodic model: atoms, orbitals, auxiliary functions (ABFs),
                      neighbour lists from true distances (lattice images included),
                      local-RI coefficient blocks T and two-centre blocks V
  build_K_blocks      production algorithm of the guide, sections 5.5 and 6.4 (real space, atom blocks, four terms)
  brute_force_K       the definition, eq. (3) with eqs. (8) and (9), summed term by term (independent code path)
  build_K_gamma       folded Gamma-only variant on dense matrices
  K_of_k, exchange_energy_blocks      eqs. (14) and (4)
  energy_gradient_gamma   dE_x/dT and dE_x/dV at fixed density matrix, eqs. (16) and (17) (input of the forces)
  write_fixture       plain-text fixture for unit tests of the C++ implementation
  integral_checks     two-/three-centre integral formulas, eqs. (11)-(13), against analytic Gaussians (SciPy)
  write_kernel_fixture    the four kernels of section 5.6 in reciprocal space, evaluated with 50 digits (mpmath)

Usage
  python3 exx_reference.py                # self-test of the algebra (NumPy only)
  python3 exx_reference.py --integrals    # integral formulas (needs SciPy)
  python3 exx_reference.py --dump FILE    # write a fixture for the C++ unit tests
  python3 exx_reference.py --kernels FILE # write the kernel fixture of the C++ unit tests (needs mpmath)

Conventions (identical to the guide)
  site            (atom, R), R an integer triple; position tau_atom + R . lattice
  D(x, y, R)      density-matrix block  D_{mu in (x,0), nu in (y,R)}  (one spin channel,
                  occupations between 0 and 1);  D(x,y,R) = D(y,x,-R)^dagger
  Bloch sums      psi_jk = sum_R exp(+2 pi i k.R) sum_mu c_{mu j}(k) phi_mu(r - tau - R)
                  D(x,y,R) = sum_k w_k exp(-2 pi i k.R) [c f c^dagger]_{xy} / sum_k w_k
                  M(k)     = sum_R exp(+2 pi i k.R) M(R)          (k in fractional coordinates)
  T[a][i][P,mu,lam]   local-RI coefficient: ABF P and orbital mu on atom a, orbital lam on the
                  i-th C-neighbour site of a (i = 0 is the atom itself).  The on-site block
                  holds one half of the symmetric on-site coefficient.
                  Full coefficient of ABF P on site A:  C^P = E_A T_P + T_P^T E_A^T
  V[a][i][P,Q]    (P|v|Q): P on atom a, Q on the i-th V-neighbour site of a
  K               K = -alpha * sum_{PQ} V_PQ C^P D C^Q,   E_x = (1/2) Tr[D K]   (per spin channel)
"""
import argparse
import itertools
import sys
from collections import defaultdict

import numpy as np

ZERO = (0, 0, 0)


def add(R, S):
    return (R[0] + S[0], R[1] + S[1], R[2] + S[2])


def sub(R, S):
    return (R[0] - S[0], R[1] - S[1], R[2] - S[2])


def neg(R):
    return (-R[0], -R[1], -R[2])


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Model:
    """Geometry, neighbour lists and local-RI data.

    lattice : (3,3), rows are the lattice vectors (bohr)
    frac    : (nat,3) fractional coordinates
    u, nP   : number of orbitals / ABFs on each atom
    rc      : orbital cut-off radius of each atom; C-neighbours: |d| <= rc[a] + rc[b]
    rv      : V-neighbours: |d| <= rv
    """

    def __init__(self, lattice, frac, u, nP, rc, rv, seed=0, random_data=True):
        self.lat = np.asarray(lattice, float)
        self.frac = np.asarray(frac, float)
        self.nat = len(self.frac)
        self.u = [int(x) for x in u]
        self.nP = [int(x) for x in nP]
        self.rc = [float(x) for x in rc]
        self.rv = float(rv)
        self.off = np.concatenate(([0], np.cumsum(self.u))).astype(int)
        self.N = int(self.off[-1])
        self.pos = self.frac @ self.lat
        self.Cnb = [self._neighbours(a, lambda b, a=a: self.rc[a] + self.rc[b]) for a in range(self.nat)]
        self.Vnb = [self._neighbours(a, lambda b: self.rv) for a in range(self.nat)]
        self.T = [[None] * len(self.Cnb[a]) for a in range(self.nat)]
        self.V = [[None] * len(self.Vnb[a]) for a in range(self.nat)]
        if random_data:
            self._random_data(np.random.default_rng(seed))

    def _neighbours(self, a, cutoff):
        rmax = max(cutoff(b) for b in range(self.nat))
        recip = np.linalg.inv(self.lat).T                  # rows b_j with a_i . b_j = delta_ij
        heights = 1.0 / np.linalg.norm(recip, axis=1)
        n = [int(np.floor(rmax / h)) + 1 for h in heights]
        found = []
        for b in range(self.nat):
            for R in itertools.product(*(range(-k, k + 1) for k in n)):
                d = self.pos[b] + np.asarray(R, float) @ self.lat - self.pos[a]
                dist = float(np.linalg.norm(d))
                if dist <= cutoff(b) + 1e-10:
                    found.append((round(dist, 9), b, R))
        found.sort()
        assert found[0][1] == a and found[0][2] == ZERO, "the atom itself must be the first neighbour"
        return [(b, R) for (_, b, R) in found]

    def _random_data(self, rng):
        for a in range(self.nat):
            for i, (b, R) in enumerate(self.Cnb[a]):
                t = 0.3 * rng.standard_normal((self.nP[a], self.u[a], self.u[b]))
                if i == 0:                                   # half of a symmetric on-site coefficient
                    t = 0.25 * (t + t.transpose(0, 2, 1))
                self.T[a][i] = t
        for a in range(self.nat):
            for i, (b, R) in enumerate(self.Vnb[a]):
                if self.V[a][i] is not None:
                    continue
                v = rng.standard_normal((self.nP[a], self.nP[b]))
                if (b, R) == (a, ZERO):
                    v = 0.5 * (v + v.T) + self.nP[a] * np.eye(self.nP[a])
                self.V[a][i] = v
                j = self.Vnb[b].index((a, neg(R)))           # (P_a|v|Q_b,R) = (Q_b|v|P_a,-R)
                self.V[b][j] = v.T.copy()

    def orb(self, a):
        return np.arange(self.off[a], self.off[a + 1])

    def stacked_T(self, a):
        """(nP, u_a, ell_a): the T blocks of atom a side by side, columns ordered as Cnb[a]"""
        return np.concatenate(self.T[a], axis=2)


def make_supercell(m, n):
    """Supercell n = (n1,n2,n3) of model m with identical T and V (translated copies).
    Returns the supercell model and the map  supercell atom -> (primitive atom, cell)."""
    cells = list(itertools.product(range(n[0]), range(n[1]), range(n[2])))
    amap = [(x, c) for c in cells for x in range(m.nat)]
    frac = np.array([(m.frac[x] + np.asarray(c)) / np.asarray(n, float) for (x, c) in amap])
    lat = m.lat * np.asarray(n, float)[:, None]
    sm = Model(lat, frac, [m.u[x] for (x, c) in amap], [m.nP[x] for (x, c) in amap],
               [m.rc[x] for (x, c) in amap], m.rv, random_data=False)

    def prim_R(c, c2, RS):                                  # primitive lattice vector of a supercell pair
        return tuple(c2[i] - c[i] + n[i] * RS[i] for i in range(3))

    for A, (x, c) in enumerate(amap):
        for i, (B, RS) in enumerate(sm.Cnb[A]):
            y, c2 = amap[B]
            sm.T[A][i] = m.T[x][m.Cnb[x].index((y, prim_R(c, c2, RS)))].copy()
        for i, (B, RS) in enumerate(sm.Vnb[A]):
            y, c2 = amap[B]
            sm.V[A][i] = m.V[x][m.Vnb[x].index((y, prim_R(c, c2, RS)))].copy()
    return sm, amap, prim_R


# ----------------------------------------------------------------------------
# density matrices
# ----------------------------------------------------------------------------
class RandomDensity:
    """Random Hermitian D(x,y,R) of unlimited range (algebra tests only)."""

    def __init__(self, m, seed=1, complex_values=False):
        self.m, self.rng, self.cplx, self.cache = m, np.random.default_rng(seed), complex_values, {}

    def __call__(self, x, y, R):
        key = (x, y, tuple(R))
        if key not in self.cache:
            g = self.rng.standard_normal((self.m.u[x], self.m.u[y]))
            if self.cplx:
                g = g + 1j * self.rng.standard_normal(g.shape)
            if x == y and key[2] == ZERO:
                g = 0.5 * (g + g.conj().T)
            self.cache[key] = g
            self.cache[(y, x, neg(key[2]))] = g.conj().T.copy()
        return self.cache[key]


class BlochDensity:
    """D(x,y,R) = sum_k w_k exp(-2 pi i k.R) P(k)_{xy} / sum_k w_k,  P(k) = c(k) f(k) c(k)^dagger.

    The phase is evaluated with the actual k points for every R: a shifted mesh makes D
    'twisted' periodic, so D must not be wrapped into the Born-von Karman cell.
    take_real=True reproduces the OpenMX convention for a list that holds one member of
    each (k, -k) pair with weight 2: the partner contributes the complex conjugate."""

    def __init__(self, m, kpts, weights, coeffs, occs, take_real=False):
        self.m = m
        self.k = np.asarray(kpts, float)
        self.w = np.asarray(weights, float)
        self.P = [(c * f) @ c.conj().T for c, f in zip(coeffs, occs)]
        self.take_real = take_real

    def __call__(self, x, y, R):
        ph = np.exp(-2j * np.pi * (self.k @ np.asarray(R, float)))
        rx, ry = self.m.orb(x), self.m.orb(y)
        blk = sum(w * p * P[np.ix_(rx, ry)] for w, p, P in zip(self.w, ph, self.P)) / self.w.sum()
        return blk.real.copy() if self.take_real else blk


class GammaDensity:
    """Gamma-only cell: D(x,y,R) is the same block of the dense matrix for every R."""

    def __init__(self, m, dense):
        self.m, self.dense = m, dense

    def __call__(self, x, y, R):
        return self.dense[np.ix_(self.m.orb(x), self.m.orb(y))]


# ----------------------------------------------------------------------------
# production algorithm: real space, atom blocks, four terms  (guide, sections 5.5 and 6.4)
# ----------------------------------------------------------------------------
def build_K_blocks(m, D, alpha):
    """K(x,y,R) blocks from D(x,y,R).  Loop over the ABF-centre atoms a of the unit cell.

    A = (a,0); loc(A) = C-neighbour sites of A; B runs over the V-neighbour sites of A;
    S = union of loc(B).  With  G_P = T_P D[loc(A), S],  W_{P,B} = sum_{Q in B} V_PQ T_Q:
        (i)   rows A,      cols B      :  sum_P G_P[:, loc(B)] W_{P,B}^T
        (ii)  rows A,      cols loc(B) :  sum_P G_P[:, B] W_{P,B}
        (iii) rows loc(A), cols B      :  sum_P T_P^T (D[A, loc(B)] W_{P,B}^T)
        (iv)  rows loc(A), cols S      :  sum_P T_P^T R_P,   R_P[:, loc(B)] += D[A, B] W_{P,B}
    Every block is accumulated into K(row atom, col atom, R_col - R_row); the result is
    multiplied by -alpha."""
    K = defaultdict(lambda: 0.0)
    for a in range(m.nat):
        loc = m.Cnb[a]
        S, Sidx = [], {}
        for (b, Rb) in m.Vnb[a]:
            for (c, Rc) in m.Cnb[b]:
                s = (c, add(Rb, Rc))
                if s not in Sidx:
                    Sidx[s] = len(S)
                    S.append(s)
        rowptr = np.concatenate(([0], np.cumsum([m.u[c] for (c, _) in loc]))).astype(int)
        colptr = np.concatenate(([0], np.cumsum([m.u[y] for (y, _) in S]))).astype(int)
        first = D(loc[0][0], S[0][0], sub(S[0][1], loc[0][1]))
        Dloc = np.zeros((rowptr[-1], colptr[-1]), dtype=np.result_type(first.dtype, float))
        for i, (c, Rc) in enumerate(loc):
            for j, (y, Ry) in enumerate(S):
                Dloc[rowptr[i]:rowptr[i + 1], colptr[j]:colptr[j + 1]] = D(c, y, sub(Ry, Rc))
        Tst = m.stacked_T(a)                                  # (nP, u_a, ell_a)
        nP, ua = m.nP[a], m.u[a]
        G = Tst @ Dloc                                        # (nP, u_a, s_a)
        Da = Dloc[:ua]                                        # rows of the site A itself
        Rm = np.zeros((nP, ua, colptr[-1]), dtype=Dloc.dtype)
        for ib, (b, Rb) in enumerate(m.Vnb[a]):
            sitesB = [(c, add(Rb, Rc)) for (c, Rc) in m.Cnb[b]]           # loc(B), first entry is B
            cB = np.arange(colptr[Sidx[sitesB[0]]], colptr[Sidx[sitesB[0]] + 1])
            cL = np.concatenate([np.arange(colptr[Sidx[s]], colptr[Sidx[s] + 1]) for s in sitesB])
            lptr = np.concatenate(([0], np.cumsum([m.u[c] for (c, _) in sitesB]))).astype(int)
            W = np.einsum('pq,qul->pul', m.V[a][ib], m.stacked_T(b))        # (nP_a, u_b, ell_b)
            Wt = W.transpose(0, 2, 1)
            t1 = (G[:, :, cL] @ Wt).sum(0)                                   # (i)
            t2 = (G[:, :, cB] @ W).sum(0)                                    # (ii)
            t3 = np.einsum('pul,puv->lv', Tst, Da[:, cL] @ Wt)               # (iii)
            Rm[:, :, cL] += Da[:, cB] @ W                                    # (iv), bracket
            K[(a, b, Rb)] = K[(a, b, Rb)] + t1
            for j, (c, Rs) in enumerate(sitesB):
                K[(a, c, Rs)] = K[(a, c, Rs)] + t2[:, lptr[j]:lptr[j + 1]]
            for i, (c, Rc) in enumerate(loc):
                key = (c, b, sub(Rb, Rc))
                K[key] = K[key] + t3[rowptr[i]:rowptr[i + 1]]
        t4 = Tst.reshape(nP * ua, -1).T @ Rm.reshape(nP * ua, -1)            # (iv), back-transform
        for i, (c, Rc) in enumerate(loc):
            for j, (y, Ry) in enumerate(S):
                key = (c, y, sub(Ry, Rc))
                K[key] = K[key] + t4[rowptr[i]:rowptr[i + 1], colptr[j]:colptr[j + 1]]
    return {key: -alpha * val for key, val in K.items()}


# ----------------------------------------------------------------------------
# definition, term by term
# ----------------------------------------------------------------------------
def brute_force_K(m, D, alpha):
    """K_{mu(x,0), nu(Y)} = -alpha sum C^P[mu(x,0), lam(L)] D[lam(L), sig(S)] V[P(A), Q(B)] C^Q[sig(S), nu(Y)]
    with C^P[mu(X), lam(L)] = [X == A] T_A[P][mu, lam at L] + [L == A] T_A[P][lam, mu at X]."""
    K = defaultdict(lambda: 0.0)
    for x in range(m.nat):
        X = (x, ZERO)
        left = [(X, s, m.T[x][i]) for i, s in enumerate(m.Cnb[x])]                    # A == X
        for a in range(m.nat):                                                        # L == A
            for i, (c, Rc) in enumerate(m.Cnb[a]):
                if c == x:
                    A = (a, neg(Rc))
                    left.append((A, A, m.T[a][i].transpose(0, 2, 1)))
        for (A, L, CL) in left:
            a, Ra = A
            for ib, (b, Rrel) in enumerate(m.Vnb[a]):
                B = (b, add(Ra, Rrel))
                right = []
                for j, (c, Rc) in enumerate(m.Cnb[b]):
                    Y = (c, add(B[1], Rc))
                    right.append((B, Y, m.T[b][j]))                                   # sig on B, nu on Y
                    right.append((Y, B, m.T[b][j].transpose(0, 2, 1)))                # sig on Y, nu on B
                for (Ssite, Y, CR) in right:
                    Dls = D(L[0], Ssite[0], sub(Ssite[1], L[1]))
                    blk = np.einsum('pml,ls,pq,qsn->mn', CL, Dls, m.V[a][ib], CR, optimize=True)
                    key = (x, Y[0], Y[1])
                    K[key] = K[key] - alpha * blk
    return dict(K)


# ----------------------------------------------------------------------------
# folded Gamma-only variant
# ----------------------------------------------------------------------------
def gamma_columns(m):
    return [np.concatenate([m.orb(b) for (b, _) in m.Cnb[a]]) for a in range(m.nat)]


def build_K_gamma(m, Dd, alpha):
    """Dense K for a Gamma-only cell.  Lattice images of a neighbour fold onto the same
    orbital indices, so the scatter steps must accumulate (np.add.at)."""
    N = m.N
    K = np.zeros((N, N), dtype=Dd.dtype)
    cols = gamma_columns(m)
    Tst = [m.stacked_T(a) for a in range(m.nat)]
    for a in range(m.nat):
        ra, ca = m.orb(a), cols[a]
        nP, ua = m.nP[a], m.u[a]
        G = Tst[a] @ Dd[ca]                                   # (nP, u_a, N)
        Da = Dd[ra]
        Rm = np.zeros((nP, ua, N), dtype=Dd.dtype)
        for ib, (b, _) in enumerate(m.Vnb[a]):
            rb, cb = m.orb(b), cols[b]
            W = np.einsum('pq,qul->pul', m.V[a][ib], Tst[b])
            Wt = W.transpose(0, 2, 1)
            K[np.ix_(ra, rb)] += (G[:, :, cb] @ Wt).sum(0)                                   # (i)
            np.add.at(K, (ra[:, None], cb[None, :]), (G[:, :, rb] @ W).sum(0))               # (ii)
            np.add.at(K, (ca[:, None], rb[None, :]),
                      np.einsum('pul,puv->lv', Tst[a], Da[:, cb] @ Wt))                      # (iii)
            np.add.at(Rm, (slice(None), slice(None), cb), Da[:, rb] @ W)                     # (iv)
        np.add.at(K, ca, Tst[a].reshape(nP * ua, -1).T @ Rm.reshape(nP * ua, N))             # (iv)
    return -alpha * K


def energy_gradient_gamma(m, Dd, alpha):
    """Derivatives of E_x = (1/2) Tr[D K[D]] at fixed real symmetric D (Gamma-only cell):
         dE/dT[a][i][P,mu,lam] = -2 alpha F^P[mu, lam],   F^P = (R_P + R'_P) D[:, loc(A)]
             R_P [:, loc(B)] += D[A, B] W_{P,B}          (bracket of term iv)
             R'_P[:, B]      += D[A, loc(B)] W_{P,B}^T   (bracket of term iii)
         dE/dV[a][i][P,Q]      = -(alpha/2) Gamma_PQ,     Gamma_PQ = Tr[D C^P D C^Q]
             Gamma_PQ = sum T_Q o (D[B,A] G_P[:, loc(B)]) + sum T_P o (D[A,B] G_Q[:, loc(A)])
                        + 2 Tr[G_P[:, B] G_Q[:, A]],      G_P = T_P D[loc(A), :]
    Every stored block is treated as an independent variable."""
    cols = gamma_columns(m)
    Tst = [m.stacked_T(a) for a in range(m.nat)]
    G = [Tst[a] @ Dd[cols[a]] for a in range(m.nat)]
    dT, dV = [], []
    for a in range(m.nat):
        ra, ca = m.orb(a), cols[a]
        Da = Dd[ra]
        Rs = np.zeros((m.nP[a], m.u[a], m.N))
        dVa = []
        for ib, (b, _) in enumerate(m.Vnb[a]):
            rb, cb = m.orb(b), cols[b]
            W = np.einsum('pq,qul->pul', m.V[a][ib], Tst[b])
            np.add.at(Rs, (slice(None), slice(None), cb), Da[:, rb] @ W)
            Rs[:, :, rb] += Da[:, cb] @ W.transpose(0, 2, 1)
            H1 = np.einsum('ji,pil->pjl', Dd[np.ix_(rb, ra)], G[a][:, :, cb])
            H2 = np.einsum('ij,qjl->qil', Dd[np.ix_(ra, rb)], G[b][:, :, ca])
            gam = (np.einsum('qjl,pjl->pq', Tst[b], H1) + np.einsum('pil,qil->pq', Tst[a], H2)
                   + 2.0 * np.einsum('pij,qji->pq', G[a][:, :, rb], G[b][:, :, ra]))
            dVa.append(-0.5 * alpha * gam)
        F = -2.0 * alpha * (Rs @ Dd[:, ca])                   # (nP, u_a, ell_a)
        ptr = np.concatenate(([0], np.cumsum([m.u[c] for (c, _) in m.Cnb[a]]))).astype(int)
        dT.append([F[:, :, ptr[i]:ptr[i + 1]] for i in range(len(m.Cnb[a]))])
        dV.append(dVa)
    return dT, dV


# ----------------------------------------------------------------------------
# k space, energy, comparison helpers
# ----------------------------------------------------------------------------
def K_of_k(m, K, k):
    """K(k) = sum_R exp(+2 pi i k.R) K(R), dense N x N"""
    out = np.zeros((m.N, m.N), complex)
    for (x, y, R), blk in K.items():
        out[np.ix_(m.orb(x), m.orb(y))] += np.exp(2j * np.pi * np.dot(k, R)) * blk
    return out


def exchange_energy_blocks(D, K):
    """E_x per unit cell and spin channel = (1/2) sum_R sum_{mu nu} conj(D_{mu nu}(R)) K_{mu nu}(R)"""
    return 0.5 * sum(np.sum(np.conj(D(x, y, R)) * blk) for (x, y, R), blk in K.items()).real


def max_diff(K1, K2):
    d = 0.0
    for key in set(K1) | set(K2):
        a, b = K1.get(key), K2.get(key)
        d = max(d, np.abs((0.0 if a is None else a) - (0.0 if b is None else b)).max())
    return d


def hermiticity_error(K):
    return max(np.abs(blk - K[(y, x, neg(R))].conj().T).max() if (y, x, neg(R)) in K else np.abs(blk).max()
               for (x, y, R), blk in K.items())


def fold_gamma(m, K):
    """sum_R K(x,y,R) as a dense matrix (Gamma-only cell)"""
    out = None
    for (x, y, R), blk in K.items():
        if out is None:
            out = np.zeros((m.N, m.N), dtype=blk.dtype)
        out[np.ix_(m.orb(x), m.orb(y))] += blk
    return out


# ----------------------------------------------------------------------------
# fixture for the C++ unit tests
# ----------------------------------------------------------------------------
def write_fixture(path, m, D, K, alpha):
    """Plain text, whitespace separated, row-major arrays, 17 significant digits.

    exx_fixture 1 / alpha / lattice (9) / natoms / per atom: u nP frac(3)
    cnb a n: n lines 'b R1 R2 R3';   vnb a n: likewise
    T a i nP u_a u_b: values;        V a i nP_a nP_b: values
    D n: n records 'x y R1 R2 R3 rows cols' + values (every block read by the algorithm)
    K n: n records, same layout;     energy: E_x of exchange_energy_blocks"""
    needed = {}

    class Recorder:
        def __call__(self, x, y, R):
            needed[(x, y, tuple(R))] = D(x, y, R)
            return needed[(x, y, tuple(R))]

    Kcheck = build_K_blocks(m, Recorder(), alpha)
    assert max_diff(K, Kcheck) < 1e-12
    fmt = lambda arr: ' '.join('%.17g' % v for v in np.asarray(arr, float).ravel())
    with open(path, 'w') as f:
        f.write('exx_fixture 1\nalpha %.17g\nlattice %s\nnatoms %d\n' % (alpha, fmt(m.lat), m.nat))
        for a in range(m.nat):
            f.write('atom %d %d %d %s\n' % (a, m.u[a], m.nP[a], fmt(m.frac[a])))
        for name, lists in (('cnb', m.Cnb), ('vnb', m.Vnb)):
            for a in range(m.nat):
                f.write('%s %d %d\n' % (name, a, len(lists[a])))
                for (b, R) in lists[a]:
                    f.write('%d %d %d %d\n' % ((b,) + tuple(R)))
        for name, data in (('T', m.T), ('V', m.V)):
            for a in range(m.nat):
                for i, blk in enumerate(data[a]):
                    f.write('%s %d %d %s\n%s\n' % (name, a, i, ' '.join(map(str, blk.shape)), fmt(blk)))
        for name, blocks in (('D', needed), ('K', K)):
            f.write('%s %d\n' % (name, len(blocks)))
            for (x, y, R) in sorted(blocks):
                blk = blocks[(x, y, R)]
                f.write('%d %d %d %d %d %d %d\n%s\n' % ((x, y) + tuple(R) + blk.shape + (fmt(blk),)))
        f.write('energy %.17g\n' % exchange_energy_blocks(D, K))


# ----------------------------------------------------------------------------
# self-test of the algebra
# ----------------------------------------------------------------------------
def default_model(seed=3):
    """Two atoms with different numbers of orbitals and ABFs in a skewed cell.  The cell is
    smaller than the cut-offs, so several images of one atom appear in a neighbour list."""
    lat = [[3.1, 0.0, 0.0], [0.7, 2.9, 0.0], [0.2, -0.3, 8.0]]
    frac = [[0.05, 0.10, 0.0], [0.55, 0.40, 0.12]]
    return Model(lat, frac, u=[3, 2], nP=[2, 3], rc=[1.6, 1.3], rv=3.4, seed=seed)


def tr_mesh(n, shift, rng, m, nocc):
    """k mesh closed under k -> -k with coefficients c(-k) = conj(c(k)); full list, weight 1"""
    pts = [np.array([(i + shift[0]) / n[0], (j + shift[1]) / n[1], (l + shift[2]) / n[2]])
           for i in range(n[0]) for j in range(n[1]) for l in range(n[2])]
    pts = [p - np.round(p) for p in pts]
    coeff, done = [None] * len(pts), {}
    for i, p in enumerate(pts):
        key = tuple(np.round(p, 9))
        mkey = tuple(np.round(-p - np.round(-p - 1e-12), 9))
        if mkey in done:
            coeff[i] = coeff[done[mkey]].conj()
        else:
            z = rng.standard_normal((m.N, nocc)) + 1j * rng.standard_normal((m.N, nocc))
            if np.allclose(key, mkey):
                z = z.real
            coeff[i] = np.linalg.qr(z)[0]
        done[key] = i
    return np.array(pts), coeff


def self_test():
    alpha, tol, ok = 0.25, 1e-11, True

    def report(name, value, limit=tol):
        nonlocal ok
        good = value < limit
        ok = ok and good
        print('%-74s %9.2e  %s' % (name, value, 'ok' if good else 'FAILED'))

    m = default_model()
    print('model: %d atoms, %d orbitals, C-neighbours %s, V-neighbours %s'
          % (m.nat, m.N, [len(x) for x in m.Cnb], [len(x) for x in m.Vnb]))

    # T1/T2: block algorithm against the definition (real and complex Hermitian D)
    for cplx in (False, True):
        D = RandomDensity(m, seed=5, complex_values=cplx)
        Kb, Kr = build_K_blocks(m, D, alpha), brute_force_K(m, D, alpha)
        tag = 'complex' if cplx else 'real'
        report('T1 block algorithm vs. definition (%s D)' % tag, max_diff(Kb, Kr))
        report('T2 K(x,y,R) = K(y,x,-R)^dagger (%s D)' % tag, hermiticity_error(Kb))

    # T3: the same system described as a supercell
    D = RandomDensity(m, seed=8)
    Kp = build_K_blocks(m, D, alpha)
    n = (2, 2, 1)
    sm, amap, prim_R = make_supercell(m, n)
    Ds = lambda X, Y, RS: D(amap[X][0], amap[Y][0], prim_R(amap[X][1], amap[Y][1], RS))
    Ks = build_K_blocks(sm, Ds, alpha)
    err = 0.0
    for (X, Y, RS), blk in Ks.items():
        ref = Kp.get((amap[X][0], amap[Y][0], prim_R(amap[X][1], amap[Y][1], RS)))
        err = max(err, np.abs(blk - (0.0 if ref is None else ref)).max())
    report('T3 supercell (2,2,1) blocks vs. primitive-cell blocks', err)

    # T4/T5: Bloch densities; folded Gamma algorithm; energy in real and reciprocal space
    rng = np.random.default_rng(21)
    nocc = 2
    for shift, label in (((0.0, 0.0, 0.0), 'unshifted mesh'), ((0.5, 0.5, 0.0), 'half-shifted mesh')):
        kp, cf = tr_mesh(n, shift, rng, m, nocc)
        occ = [np.array([1.0, 0.6])] * len(kp)
        Db = BlochDensity(m, kp, np.ones(len(kp)), cf, occ)
        imag = max(np.abs(Db(x, y, R).imag).max() for x in range(m.nat) for y in range(m.nat)
                   for R in ((0, 0, 0), (1, 0, 0), (-2, 1, 0)))
        report('   D(R) is real for a k mesh closed under inversion (%s)' % label, imag)
        Dr = BlochDensity(m, kp, np.ones(len(kp)), cf, occ, take_real=True)
        Kb = build_K_blocks(m, Dr, alpha)
        e_real = exchange_energy_blocks(Dr, Kb)
        e_k, herm = 0.0, 0.0
        for k, P in zip(kp, Db.P):
            Kk = K_of_k(m, Kb, k)
            herm = max(herm, np.abs(Kk - Kk.conj().T).max())
            e_k += 0.5 * np.trace(P @ Kk).real / len(kp)
        report('T5 K(k) Hermitian (%s)' % label, herm)
        report('T5 E_x: real-space sum vs. (1/2N_k) sum_k Tr[P(k)K(k)] (%s)' % label, abs(e_real - e_k))
        if shift == (0.0, 0.0, 0.0):
            Dd = np.zeros((sm.N, sm.N))
            for X in range(sm.nat):
                for Y in range(sm.nat):
                    Dd[np.ix_(sm.orb(X), sm.orb(Y))] = Dr(amap[X][0], amap[Y][0],
                                                          prim_R(amap[X][1], amap[Y][1], ZERO))
            Kg = build_K_gamma(sm, Dd, alpha)
            Kf = fold_gamma(sm, build_K_blocks(sm, GammaDensity(sm, Dd), alpha))
            report('T4 folded Gamma algorithm vs. sum_R K(R) of the supercell', np.abs(Kg - Kf).max())
            report('T4 dense K symmetric', np.abs(Kg - Kg.T).max())
            e_super = 0.5 * np.sum(Dd * Kg)
            report('T5 E_x(supercell, Gamma) = N_cell * E_x(primitive cell, k mesh)',
                   abs(e_super - len(kp) * e_real))
            # T6: derivatives of E_x with respect to the T and V blocks
            dT, dV = energy_gradient_gamma(sm, Dd, alpha)
            energy = lambda: 0.5 * np.sum(Dd * build_K_gamma(sm, Dd, alpha))
            h = 1e-5
            for name, data, grad in (('T', sm.T, dT), ('V', sm.V, dV)):
                saved = [[blk.copy() for blk in row] for row in data]
                pert = [[rng.standard_normal(blk.shape) for blk in row] for row in data]
                ana = sum(np.sum(g * p) for gr, pr in zip(grad, pert) for g, p in zip(gr, pr))
                vals = []
                for sgn in (+1, -1):
                    for a in range(sm.nat):
                        for i in range(len(data[a])):
                            data[a][i] = saved[a][i] + sgn * h * pert[a][i]
                    vals.append(energy())
                for a in range(sm.nat):
                    for i in range(len(data[a])):
                        data[a][i] = saved[a][i]
                num = (vals[0] - vals[1]) / (2 * h)
                report('T6 dE_x/d%s blocks vs. central difference (relative)' % name,
                       abs(ana - num) / abs(num), 1e-7)
    print('self-test', 'passed' if ok else 'FAILED')
    return ok


# ----------------------------------------------------------------------------
# integral formulas against analytic Gaussian values (SciPy)
# ----------------------------------------------------------------------------
def integral_checks():
    """(P|v|Q)_d = 8 sum_{LM} i^{lP-lQ-L} Gaunt(lP mP, lQ mQ, L M) Y_LM(d^) I_L(d),
       I_L(d) = int_0^inf kappa^2 ptil(kappa) qtil(kappa) vtil(kappa) j_L(kappa d) dkappa,
       ptil(kappa) = int_0^inf r^2 p(r) j_lP(kappa r) dr,  P(r) = p(r) Y_lm(r^), real harmonics,
       d = vector from the centre of P to the centre of Q.
       vtil: 1 (overlap), 4 pi / kappa^2 (Coulomb), 4 pi (1 - exp(-kappa^2 / 4 omega^2)) / kappa^2 (erfc),
             4 pi (1 - cos(kappa Rc)) / kappa^2 (Coulomb truncated at Rc).
       Three-centre integrals with two functions on one atom use the product of the orbital
       and the Coulomb potential of the ABF as one on-site function."""
    from scipy.special import erf, spherical_jn

    ok = True

    def Y(l, mm, v):
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        r2 = x * x + y * y + z * z
        if l == 0:
            return np.full(r2.shape, 0.5 / np.sqrt(np.pi))
        if l == 1:
            return np.sqrt(3 / (4 * np.pi)) * {-1: y, 0: z, 1: x}[mm] / np.sqrt(r2)
        c = np.sqrt(15 / (4 * np.pi))
        return {-2: c * x * y, -1: c * y * z, 0: np.sqrt(5 / (16 * np.pi)) * (3 * z * z - r2),
                1: c * x * z, 2: 0.5 * c * (x * x - y * y)}[mm] / r2

    xs, ws = np.polynomial.legendre.leggauss(24)
    ph = 2 * np.pi * np.arange(48) / 48
    st = np.sqrt(1 - xs ** 2)
    dirs = np.stack([np.outer(st, np.cos(ph)), np.outer(st, np.sin(ph)), np.outer(xs, np.ones(48))], axis=-1)
    wang = np.outer(ws, np.full(48, 2 * np.pi / 48))
    gaunt = lambda l1, m1, l2, m2, L, M: np.sum(wang * Y(l1, m1, dirs) * Y(l2, m2, dirs) * Y(L, M, dirs))

    r = np.linspace(0.0, 12.0, 1201)
    q = np.linspace(0.0, 30.0, 3001)
    dr, dq = r[1] - r[0], q[1] - q[0]
    trapz = lambda y, h: h * (np.sum(y, axis=-1) - 0.5 * (y[..., 0] + y[..., -1]))
    bessel = {l: spherical_jn(l, np.outer(q, r)) for l in (0, 1)}
    ftil = lambda rad, l: trapz(r * r * rad * bessel[l], dr)

    def twocenter(pr, lP, mP, qr, lQ, mQ, q2v, d):
        dist = np.linalg.norm(d)
        pt, qt = ftil(pr, lP), ftil(qr, lQ)
        val = 0.0
        for L in range(abs(lP - lQ), lP + lQ + 1, 2):
            IL = trapz(pt * qt * q2v(q) * spherical_jn(L, q * dist), dq)
            phase = (1j) ** (lP - lQ - L)
            for M in range(-L, L + 1):
                g = gaunt(lP, mP, lQ, mQ, L, M)
                if abs(g) > 1e-12:
                    val += 8.0 * phase.real * g * Y(L, M, np.asarray(d, float)) * IL
        return float(val)

    def report(name, num, ref, limit):
        nonlocal ok
        good = abs(num - ref) < limit
        ok = ok and good
        print('%-46s formula %+.10f  analytic %+.10f  diff %.1e  %s'
              % (name, num, ref, abs(num - ref), 'ok' if good else 'FAILED'))

    a, b, om, Rc = 0.9, 0.6, 0.11, 3.0
    mu = a * b / (a + b)
    mue = mu * om * om / (mu + om * om)
    pref = (np.pi ** 2 / (a * b)) ** 1.5
    kernels = {                                               # kappa^2 * vtil(kappa), F(d) for two s Gaussians
        'coulomb': (lambda k: 4 * np.pi + 0 * k, lambda d: erf(np.sqrt(mu) * d) / d),
        'erfc': (lambda k: 4 * np.pi * (1 - np.exp(-k * k / (4 * om * om))),
                 lambda d: (erf(np.sqrt(mu) * d) - erf(np.sqrt(mue) * d)) / d),
        'truncated': (lambda k: 4 * np.pi * (1 - np.cos(k * Rc)),
                      lambda d: (2 * erf(np.sqrt(mu) * d) + erf(np.sqrt(mu) * (Rc - d))
                                 - erf(np.sqrt(mu) * (Rc + d))) / (2 * d)),
    }
    ga, gb = np.exp(-a * r * r), np.exp(-b * r * r)
    pa, pb = r * ga, r * gb
    d = np.array([0.7, -0.4, 1.3])
    dist, h = np.linalg.norm(d), 1e-4
    for name, (q2v, F) in kernels.items():
        dF = (F(dist + h) - F(dist - h)) / (2 * h)
        d2F = (F(dist + h) - 2 * F(dist) + F(dist - h)) / h ** 2
        report(name + ' s -s', twocenter(ga, 0, 0, gb, 0, 0, q2v, d), pref * F(dist) / (4 * np.pi), 1e-9)
        report(name + ' pz-s', twocenter(pa, 1, 0, gb, 0, 0, q2v, d),
               np.sqrt(3) / (4 * np.pi) * pref / (2 * a) * (-d[2] / dist) * dF, 2e-7)
        report(name + ' s -pz', twocenter(ga, 0, 0, pb, 1, 0, q2v, d),
               np.sqrt(3) / (4 * np.pi) * pref / (2 * b) * (d[2] / dist) * dF, 2e-7)
        hxz = (d2F - dF / dist) * d[0] * d[2] / dist ** 2
        report(name + ' px-pz', twocenter(pa, 1, 1, pb, 1, 0, q2v, d),
               3 / (4 * np.pi) * pref / (4 * a * b) * (-hxz), 2e-7)
        hzz = (d2F - dF / dist) * d[2] ** 2 / dist ** 2 + dF / dist
        report(name + ' pz-pz', twocenter(pa, 1, 0, pb, 1, 0, q2v, d),
               3 / (4 * np.pi) * pref / (4 * a * b) * (-hzz), 2e-7)

    # three-centre Coulomb integrals (phi_mu phi_lam | P) with mu and P on the same atom
    c, p = 0.5, 0.8                                           # exponents of phi_lam and of the ABF
    gp, gc = np.exp(-p * r * r), np.exp(-c * r * r)

    def cumint(f):                                            # cumulative integral from r = 0, error O(dr^4)
        df = np.gradient(f, dr, edge_order=2)
        return np.concatenate(([0.0], np.cumsum(0.5 * dr * (f[1:] + f[:-1])))) - dr * dr / 12.0 * (df - df[0])

    inner, tail = cumint(gp * r * r), cumint(gp * r)
    outer = tail[-1] - tail
    # l = 0 potential: w(r) = 4 pi [ (1/r) int_0^r p r'^2 dr' + int_r^inf p r' dr' ]
    w = 4 * np.pi * (np.divide(inner, r, out=np.zeros_like(r), where=r > 0) + outer)
    exact = (np.pi / p) ** 1.5 * np.divide(erf(np.sqrt(p) * r), r, out=np.full_like(r, 2 * np.sqrt(p / np.pi)),
                                           where=r > 0)
    report('radial Poisson formula, max error', np.abs(w - exact).max(), 0.0, 1e-6)

    def plain3c(A):                                           # s Gaussians: a at A, c at d, p at the origin
        gam = a + c
        Rp = (a * A + c * d) / gam
        x = gam * p / (gam + p) * (Rp @ Rp)
        F0 = 0.5 * np.sqrt(np.pi / x) * erf(np.sqrt(x)) if x > 1e-12 else 1.0
        return 2 * np.pi ** 2.5 / (gam * p * np.sqrt(gam + p)) * np.exp(-a * c / gam * ((A - d) @ (A - d))) * F0

    overlap = lambda k: k * k
    num = twocenter(ga * w / np.sqrt(4 * np.pi), 0, 0, gc, 0, 0, overlap, d)
    report('three-centre (s s | s)', num, plain3c(np.zeros(3)) / (4 * np.pi) ** 1.5, 1e-8)
    ez = np.array([0.0, 0.0, 1.0])
    dz = (plain3c(h * ez) - plain3c(-h * ez)) / (2 * h)
    num = twocenter(r * ga * w / np.sqrt(4 * np.pi), 1, 0, gc, 0, 0, overlap, d)
    report('three-centre (pz s | s)', num, np.sqrt(3 / (4 * np.pi)) / (4 * np.pi) * dz / (2 * a), 2e-7)
    print('integral checks', 'passed' if ok else 'FAILED')
    return ok


# ----------------------------------------------------------------------------
# kernels in reciprocal space (mpmath)
# ----------------------------------------------------------------------------
def write_kernel_fixture(path):
    """v~(kappa) and kappa^2 v~(kappa) of the four kernels of section 5.6, evaluated with
       50 digits and rounded to the nearest double.  The parameters and the points are the
       doubles that the C++ test reads back, so the reference is the exact function of the
       same arguments.

       The truncated Coulomb kernel is 8 pi sin(kappa Rc / 2)^2 / kappa^2.  Its relative
       condition number is 2 y cot(y) - 2 with y = kappa Rc / 2, so the points keep away
       from the zeros of the sine: |y cot(y)| stays below 100 in this table."""
    import mpmath as mp

    mp.mp.dps = 50
    pi = mp.pi

    def value(kind, par, k):
        if kind == 'overlap':
            return mp.mpf(1)
        if kind == 'coulomb':
            return 4 * pi / k ** 2
        if kind == 'short_range':
            return 4 * pi * (1 - mp.exp(-k ** 2 / (4 * par ** 2))) / k ** 2
        return 4 * pi * (1 - mp.cos(k * par)) / k ** 2

    def limit(kind, par):
        if kind == 'overlap':
            return mp.mpf(1)
        if kind == 'short_range':
            return pi / par ** 2
        return 2 * pi * par ** 2

    points = [1e-12, 1e-8, 1e-5, 1e-3, 0.01, 0.1, 0.3, 1.0, 2.5, 7.0, 20.0, 60.0]
    cases = [('overlap', 0.0, points), ('coulomb', 0.0, points)]
    cases += [('short_range', om, points) for om in (0.11, 0.106, 0.3)]
    more = points + [0.03, 0.2, 0.45, 0.7, 1.6, 3.9, 11.0, 33.0]
    well = lambda par, k: abs(mp.mpf(k) * par / 2 * mp.cot(mp.mpf(k) * par / 2)) < 100
    cases += [('truncated_coulomb', rc, sorted(k for k in more if well(rc, k))) for rc in (3.0, 12.5, 31.7)]
    out = ['exx_kernel_fixture 1',
           '# written by tools/exx_reference.py --kernels (mpmath %s, %d digits)' % (mp.__version__, mp.mp.dps),
           '# limit <kernel> <parameter> <v~ at kappa = 0>',
           '# point <kernel> <parameter> <kappa> <v~(kappa)> <kappa^2 v~(kappa)>']
    count = 0
    for kind, par, ks in cases:
        p = mp.mpf(par)
        if kind != 'coulomb':
            out.append('limit %s %r %r' % (kind, par, float(limit(kind, p))))
        for k in ks:
            kk = mp.mpf(k)
            v = value(kind, p, kk)
            out.append('point %s %r %r %r %r' % (kind, par, k, float(v), float(kk * kk * v)))
            count += 1
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--integrals', action='store_true', help='check the integral formulas (needs SciPy)')
    ap.add_argument('--dump', metavar='FILE', help='write a fixture for the C++ unit tests')
    ap.add_argument('--kernels', metavar='FILE', help='write the kernel fixture for the C++ unit tests (needs mpmath)')
    args = ap.parse_args()
    if args.integrals:
        return 0 if integral_checks() else 1
    if args.kernels:
        print('wrote', args.kernels, 'with', write_kernel_fixture(args.kernels), 'points')
        return 0
    if args.dump:
        m = default_model()
        D = RandomDensity(m, seed=5)
        K = build_K_blocks(m, D, 0.25)
        write_fixture(args.dump, m, D, K, 0.25)
        print('wrote', args.dump, 'with', len(K), 'K blocks')
        return 0
    return 0 if self_test() else 1


if __name__ == '__main__':
    sys.exit(main())
