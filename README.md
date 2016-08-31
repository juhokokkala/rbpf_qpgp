## Rao-Blackwellized particle filters for quasiperiodic Gaussian processes

A Rao-Blackwellized particle filter for a model with Poisson-distributed observations whose log-intensity follows a quasiperiodic Gaussian process.

Documentation in http://www.juhokokkala.fi/blog/posts/rao-blackwellized-particle-filters-for-quasiperiodic-gaussian-processes/

## Requirements
NumPy, GPy. matplotlib for plotting. Python 3.5 due to the matrix multiplication operator.

## Files

rbpf.py -- contains the filter and the Particle independent Metropolis-Hastings sampler.
test.py -- script for running the experiment in the blog post.
resampling.py -- resampling algorithms (c) Roger R Labbe Jr.

## License information

Copyright (c) Juho Kokkala 2016 (except the file resampling.py). All files in this repository are licensed under the MIT License. See the file LICENSE or http://opensource.org/licenses/MIT. 

The file resampling.py is Copyright (c) Roger R Labbe Jr. 2015,  taken from the filterpy package, http://github.com/ - also licensed under the MIT License.