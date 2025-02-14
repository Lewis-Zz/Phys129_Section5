import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# (i) Near-Degenerate Bose System: Numerical Simulation
# --------------------------------------------------------------------


# --------------- System Parameters ---------------
kB       = 1.0           # Boltzmann const (set =1 for simplicity)
N_target = 1e5           # Total number of bosons
T_min, T_max = 0.5, 5.0  # Temperature range (arbitrary units)
nT       = 100           # Number of temperature points

# Example: "near-degenerate" ground states + a few excited states
# energies[i] = energy of level i,  degeneracies[i] = degeneracy of that level
energies     = np.array([0.0,      0.0005,   0.001,    0.01,   0.1,   0.5,   1.0])
degeneracies = np.array([100,      50,       20,       10,     5,     3,     2])
# The first entry has 100 states with energy = 0, so a "near-degenerate" ground manifold

# --------------- Define Helper Functions ---------------

def bose_occupation(eps, mu, T):
    """
    Returns the Bose-Einstein occupation number for a single-particle
    energy eps, chemical potential mu, at temperature T.
    occupation = 1 / (exp((eps - mu)/(kB*T)) - 1), assuming kB=1 in units.
    """
    x = (eps - mu)/(kB*T)
    # To avoid division by zero or negative argument, handle large x
    if x > 30:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)

def total_particle_number(mu, T):
    """
    Given chemical potential mu and temperature T, sum over
    degeneracies * bose_occupation to get <N>.
    """
    return np.sum(degeneracies * [bose_occupation(e, mu, T) for e in energies])

def total_energy(mu, T):
    """
    Returns <E> = sum_{i} [deg_i * eps_i * occupation(eps_i, mu, T)].
    """
    return np.sum(degeneracies * energies *
                  [bose_occupation(e, mu, T) for e in energies])

def find_mu_for_N(N, T):
    """
    Solve for mu in ( -infinity, min(eps_i) ) so that <N> = N_target.
    We'll do a simple bisection over mu in [mu_low, mu_high].
    Since we must have mu < lowest energy for a normal Bose gas,
    set mu_high = 0 and mu_low = -some large number.
    """
    mu_low = -50.0
    mu_high = 0.0
    # Bisection parameters
    for _ in range(200):
        mu_mid = 0.5*(mu_low + mu_high)
        N_mid = total_particle_number(mu_mid, T)
        if N_mid > N:
            mu_high = mu_mid
        else:
            mu_low  = mu_mid
    return 0.5*(mu_low + mu_high)

# --------------- Arrays to Store Quantities vs T ---------------
T_vals     = np.linspace(T_min, T_max, nT)
mu_vals    = np.zeros_like(T_vals)
n0_vals    = np.zeros_like(T_vals)   # Ground-state occupation
dn0dT_vals = np.zeros_like(T_vals)   # d(<n0>)/dT
Cv_vals    = np.zeros_like(T_vals)   # Specific heat
E_vals     = np.zeros_like(T_vals)   # <E> for each T

# --------------- Compute for each temperature ---------------
for i, T in enumerate(T_vals):
    # Solve for mu so that <N> = N_target
    mu_T = find_mu_for_N(N_target, T)
    mu_vals[i] = mu_T


    n0_vals[i] = degeneracies[0] * bose_occupation(energies[0], mu_T, T)

    # Total energy <E>
    E_vals[i] = total_energy(mu_T, T)

# --------------- Numerical Derivatives vs T ---------------
# We'll do finite differences for dn0/dT and dE/dT:
for i in range(1, nT-1):
    dT = T_vals[i+1] - T_vals[i-1]
    dn0dT_vals[i] = (n0_vals[i+1] - n0_vals[i-1]) / dT
    dEdT = (E_vals[i+1] - E_vals[i-1]) / dT
    # Specific heat C_v = d<E>/dT (assuming volume is constant, kB=1 in these units)
    Cv_vals[i] = dEdT

# --------------- Plot Results ---------------
plt.figure(figsize=(10,6))

# 1) negative chemical potential
plt.subplot(2,3,1)
plt.plot(T_vals, -mu_vals, 'b-o', ms=3)
plt.xlabel('T')
plt.ylabel(r'$-\mu$')
plt.title('Negative Chemical Potential')

# 2) ground state occupation
plt.subplot(2,3,2)
plt.plot(T_vals, n0_vals, 'r-o', ms=3)
plt.xlabel('T')
plt.ylabel(r'$\langle n_0 \rangle$')
plt.title('Ground-State Occupation')

# 3) log ground state occupation
plt.subplot(2,3,3)
# Avoid log(0) issues:
log_n0 = np.log(n0_vals + 1e-30)
plt.plot(T_vals, log_n0, 'g-o', ms=3)
plt.xlabel('T')
plt.ylabel(r'$\log(\langle n_0 \rangle)$')
plt.title('Log of Ground-State Occupation')

# 4) negative derivative w.r.t. T of <n0>
plt.subplot(2,3,4)
# We'll use the previously computed dn0dT_vals
# The edges will be zero because we didn't do a derivative there
neg_dn0dT = -dn0dT_vals
plt.plot(T_vals, neg_dn0dT, 'm-o', ms=3)
plt.xlabel('T')
plt.ylabel(r'$-\frac{d \langle n_0\rangle}{dT}$')
plt.title('Negative Slope of Ground-State Occupation')

# 5) Specific heat Cv
plt.subplot(2,3,5)
plt.plot(T_vals, Cv_vals, 'k-o', ms=3)
plt.xlabel('T')
plt.ylabel(r'$C_v$')
plt.title('Specific Heat')

plt.tight_layout()
plt.show()

