###############################################################################
# Copyright (C) 2016 Juho Kokkala
#
# This file is licensed under the MIT License.
###############################################################################
"""
A bootstrap Rao-Blackwellized PF for a Poisson model

The model is assumed to be of the form
x_(k+1) = A x_k  + q_k,   q_k ~ N(0,Q)
y_k = Poisson(exp(H x_k + offset))

This is a Rao--Blackwellized bootstrap particle filter - at each step
we propose H x_k from the model predicted distribution and use Kalman
filter to keep track of p(x_k | H x_1, H x_2, ..., H x_{k-1})
"""

import numpy as np
from scipy import stats
import resampling  # Resampling algorithms (c) Roger R. Labbe Jr.


def kalman_predict(m, P, A, Q):
    """Kalman filter prediction step"""
    m_p = A @ m
    P_p = A @ P @ A.T + Q
    return m_p, P_p


def kalman_update(m, P, H, v, S):
    """
    Kalman filter update step for scalar measurement
    with precomputed innovation moments
    """
    K = P @ H.T * (S**-1)
    m_u = m + K @ v[None, :]
    P_u = P - S * K @ K.T

    return m_u, P_u


def rbpf(m0, P0, ys, A, Q, H, offset, N, verbose=False):
    """Rao-Blackwellized particle filter"""

    # If there is only one A/Q/H, change to list
    if not isinstance(A, list):
        A = [A] * len(ys)
    if not isinstance(Q, list):
        Q = [Q] * len(ys)
    if not isinstance(H, list):
        H = [H] * len(ys)

    # Initialize particles
    W = np.tile(1/N, N)
    ms = np.tile(m0[:, None], (1, N))
    P = P0

    Hx = np.zeros((len(ys), N))
    filtermean = np.zeros(len(ys))

    loglh = 0

    # Particle filter loop
    for k, y in enumerate(ys):
        if verbose:
            print("Particle filter running step:")
            print(k)

        # Predict
        ms, P = kalman_predict(ms, P, A[k], Q[k])

        # Update
        logW = np.log(W)

        # Compute distribution of Hx and propose Hx
        m_Hx = H[k] @ ms
        S_Hx = H[k] @ P @ H[k].T
        v_Hx = np.random.normal(0, scale=np.sqrt(S_Hx), size=N)
        Hx[k, :] = m_Hx + v_Hx

        # Update sufficient statistics
        ms, P = kalman_update(ms, P, H[k], v_Hx, S_Hx)

        # Update weights
        logW += y * (Hx[k, :] + offset) - np.exp(Hx[k, :] + offset)

        maxlogW = np.max(logW)
        W = np.exp(logW - maxlogW)
        loglh = loglh + maxlogW + np.log(np.sum(W))

        # Weight renormalization (or trick in case of numeric issue
        if not np.isnan(sum(W)):
            W = W / np.sum(W)
        else:
            W = np.tile(1/N, N)

        filtermean[k] = np.sum(W * np.exp(Hx[k, :] + offset))

        # Resample - always except the last step
        if k < (len(ys)-1):
            indices = resampling.stratified_resample(W)
            ms = ms[:, indices]
            Hx[0:(k+1), :] = Hx[0:(k+1), indices]
            W = np.tile(1/N, N)

    return Hx, loglh, W, filtermean


def pimh(m0, P0, ys, A, Q, H, offset, Nparticles, Nsamples):
    """Particle independent Metropolis-Hastings sampler"""

    # Initialization
    print("PIMH Initializing")
    Hx_star, loglh_current, W, _ = rbpf(m0, P0, ys, A, Q, H, offset,
                                        Nparticles)
    K = np.random.choice(Nparticles, p=W)
    Hx_current = Hx_star[:, K]
    Hx_mcmc = np.zeros((len(ys), Nsamples))

    # MCMC loop
    for i in range(Nsamples):
        print("PIMH Running step:")
        print(i)
        Hx_star, loglh_star, W, _ = rbpf(m0, P0, ys, A, Q, H, offset
                                         Nparticles)
        if np.random.uniform() <= np.exp(loglh_star - loglh_current):
            print("Accept!")
            K = np.random.choice(Nparticles, p=W)
            loglh_current = loglh_star
            Hx_current = Hx_star[:, K]
        Hx_mcmc[:, i] = Hx_current

    return Hx_mcmc
