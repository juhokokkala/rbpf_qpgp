###############################################################################
# Copyright (C) 2016 Juho Kokkala
#
# This file is licensed under the MIT License.
###############################################################################
"""A script for replicating the experiment."""


import GPy
import numpy as np
import GPy.models.state_space_main as ssm
from matplotlib import pyplot as plt

import rbpf

## Parameters

days = 4
slices_in_day = 159
dt = 1/288
assert(dt*slices_in_day < 1)

kernel1 = GPy.kern.StdPeriodic(1, period=1, lengthscale=0.5, variance=2)
kernel2 = GPy.kern.Matern32(1, lengthscale=10)
kernel3 = GPy.kern.Exponential(1, variance=0.15, lengthscale=0.3)
kernel4 = kernel1 * kernel2 + kernel3

mu = 0.5

## Simulating data

np.random.seed(2)  # Since I used (1) for tweaking the params

t_day = dt * np.arange(slices_in_day)
t = t_day.copy()
for d in range(1, days):
    t = np.concatenate((t, d + t_day))

Sigma = kernel4.K(t[:, None], t[:, None])

x = np.random.multivariate_normal(np.repeat(mu, t.shape[0]), Sigma)
y = np.random.poisson(np.exp(x))

## Constructing the state-space approximation

kernel1 = GPy.kern.sde_StdPeriodic(1, period=1, lengthscale=0.5, variance=2)
kernel2 = GPy.kern.sde_Matern32(1, lengthscale=10)
kernel3 = GPy.kern.sde_Exponential(1, variance=0.15, lengthscale=0.3)

kernel4 = kernel1 * kernel2 + kernel3

(F, L, Qc, H, P_inf, P0, _, _, _, _) = kernel4.sde()

CDSS = ssm.ContDescrStateSpace

dt_interday = 1 - dt * slices_in_day

A_intraday, Q_intraday, _, _, _ = CDSS.lti_sde_to_descrete(F, L, Qc, dt,
                                                           P_inf=P_inf)

A_interday, Q_interday, _, _, _ = CDSS.lti_sde_to_descrete(F, L, Qc,
                                                           dt_interday,
                                                           P_inf=P_inf)

A = days * ([A_interday] + (slices_in_day - 1) * [A_intraday])
Q = days * ([Q_interday] + (slices_in_day - 1) * [Q_intraday])

## Filtering

np.random.seed(3)

logl, lhestimate, W, filtermean = rbpf.rbpf(np.zeros(P_inf.shape[0]), P_inf,
                                            y.flatten(order='F'), A, Q, H, mu,
                                            200, verbose=True)

## MCMC

logl_mcmc = rbpf.pimh(np.zeros(P_inf.shape[0]), P_inf, y, A, Q, H, mu, 200,
                      1000)

mcmc_mean = np.mean(np.exp(logl_mcmc[:, 250:] + mu), axis=1)

## Saving results

np.savez("results.npz", logl_mcmc=logl_mcmc, logl=logl, W=W,
         filtermean=filtermean)

## Compute RMSEs

prior_pred = np.exp(mu + 0.5 * (2 + 0.3))


def rmse(prediction):
    return np.sqrt(np.mean(np.square(prediction - np.exp(x))))

print(rmse(prior_pred))
print(rmse(y))
print(rmse(filtermean))
print(rmse(mcmc_mean))

##  Plot stuff
t_ = dt * np.arange(len(t))
plt.cla()
plt.plot(t_, np.exp(x), 'k-')
plt.plot(t_, filtermean, 'r--')
plt.plot(t_, y, 'k+')
for d in range(1, days):
    plt.plot(2 * [dt * (d * slices_in_day-1)], [0, np.max(y)], 'k-')

plt.legend(['Ground truth', 'Filtering mean', 'Observations'])
plt.xlim([0, t_[-1]])
plt.ylim([0, max((np.max(y), np.max(np.exp(x))))])

plt.savefig('rbpfqpgp_filter.png')

plt.show()

##
plt.cla()


plt.plot(t_, np.exp(x), 'k-')
plt.plot(t_, mcmc_mean, 'g--')
plt.plot(t_, filtermean, 'r-.')
for d in range(1, days):
    plt.plot(2 * [dt * (d * slices_in_day - 1)], [0, np.max(y)], 'k-')

plt.yscale('log')

plt.legend(['Ground truth', 'MCMC mean', 'Filter mean'])
plt.xlim([0, t_[-1]])
plt.ylim([0, np.max(np.exp(x))])

plt.savefig('rbfqpgp_mcmc.png')

plt.show()
