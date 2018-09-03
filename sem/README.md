# README

aponteeduardo@gmail.com
copyright (C) 2015-2017

# The SERIA model

## Quick start

The [SERIA model](http://www.biorxiv.org/content/early/2017/06/08/109090)
is a formal statistical model of the probability of a 
pro- or antisaccade and its reaction time. The SEM toolbox includes an 
inference method based on the Metropolis-Hasting algorithm implemented in
MATLAB.

After installation (see below), you can run an example using

~~~~
tapas_init();
tapas_sem_flat_example_invesion();
~~~~

This will load data and estimate parameters. The data consists
of a list of trials with trial type (pro or antisaccade), the
action performed (pro or antisaccade) and the reaction time. 

You can use the file `sem/examples/tapas_sem_example_inversion.m`
as a template to run your analysis.

## The model

## Parametric distributions
Different parametric distributions can be used to model the hit time of
the units. We recommend to model the hit times with the inverse Gamma
distribution, or a combination of the Gamma and inverse Gamma functions.
Below is a table with all the available options, including the name,
the distribution of the early and late units, and the name of the
function that implements each of the models (i.e., the likelihood 
function).

| Name | Early \& inhibitory unit | Late units | Likelihood function |
|:-----:|:-----:|:-----:|:-----:|
| Gamma | Gamma | Gamma | c_seria_multi_gamma |
| Inv. Gamma | Inv. Gamma | Inv. Gamma | c_seria_multi_invgamma |
| Mixed Gamma | Inv. Gamma | Gamma | c_seria_multi_mixedgamma |
| Log. Normal | Log. Normal | Log. Normal | c_seria_multi_lognorm |
| Wald | Wald | Wald | c_seria_multi_wald |
| Later | Later | Later | c_seria_multi_later |

The [Wald distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
is the hit time distribution of a Drift-diffusion process with a single
boundary. The Later model is the distribution of the random variable \(1/X\),
where \(X\) is truncated normal distributed such that \(X>0\).

## Parameters coding
The parameters of SERIA are organized as 11x1 vector. The table below
explains the meaning of each parameter.

|\# | Meaning |
|:----:|-------|
|1 | log mean hit time early unit|
|2 | log variance hit time early unit|
|3 | log mean hit time inhibitory unit|
|4 | log variance hit time inhibitory unit|
|5 | log mean hit time anti. unit|
|6 | log variance hit time anti. unit|
|7 | log mean hit time late pro. unit|
|8 | log variance hit time late pro. unit|
|9 | log no decision time|
|10 | logit of the probability of an early outlier|
|11 | log late units delay |

All the parameters are in a scale from \(-\infty\) to \(\infty\). The
appropriate transformations are implemented internally depending on the
parametric distribution used for the hit time of the units.

### A note on the PROSA model
In the PROSA model, the assumption that late prosaccades can be generated
by a late unit is dropped. Instead, all prosaccades are early saccades.
Because the PROSA model lacks late prosaccades, it has 2 parameters less
than the SERIA model (parameters 7 and 8). The same set of parametric 
distributions are implemented for the SERIA and PROSA models according
to the table

| Name | Early \& inhibitory unit | Late units | Likelihood function |
|:-----:|:-----:|:-----:|:-----:|
| Gamma | Gamma | Gamma | c_prosa_multi_gamma |
| Inv. Gamma | Inv. Gamma | Inv. Gamma | c_prosa_multi_invgamma |
| Mixed Gamma | Inv. Gamma | Gamma | c_prosa_multi_mixedgamma |
| Log. Normal | Log. Normal | Log. Normal | c_prosa_multi_lognorm |
| Wald | Wald | Wald | c_prosa_multi_wald |
| Later | Later | Later | c_prosa_multi_later |


## Data coding
The data entered to the model is encoded as a structure with the fields
`y` and `u`. This is an structure array, in which the number of rows 
corresponds to the number of subjects.
 
The field `y` represents the responses of a subjects in terms of RT (in 
tenths of a second) and the corresponding action. The contents of each
field is represented by a vector of Nx1 trials.

| Fields `y` | Meaning | Data |
|:----------:|:-------:|:----:|
| `t`        | Reaction time | Seconds x 1/10 |
| `a`        | Action        | Pro=0, Anti=1 |

The field `u` represents experimental conditions and it has a single 
subfield `tt`, which is a vector of Nx1 trials. `u.tt` codes the condition
of the corresponding trial. Conditions should be coded by 
by integers starting from 0. 

| Fields *u* | Meaning | Data |
|:----------:|:-------:|:----:|
|*tt*       | Trial type <br> (condition) | Integer from 0 to M |

For example, if in an experiment pro- and antisaccade trials are mixed in
a single block, it is possible to code these two types of trials as 0 and 1.
Note that the consequences of coding two types of trials as two conditions
is that a different set of parameters will be initialized for each condition.

## Constraints
It is possible to enforce constraints on the parameters of a model
across conditions of a single subject using a projection matrix. 
This matrix *J* should have *M* times 11 rows and *K* columns, 
where *M* is the number of condition and *K* is the number of free parameters.

As an example, imagine that we want to enforce that the no decision time, 
the probability of an early outlier and the delay of the late units
are shared across two conditions (for example, across pro and antisaccade 
trials). This is implemented by enforcing that in the product of a vector *v* 
of dimensionality 11x2-3 with matrix *J*, the entries 9 to eleven are 
equal to the entries 20 to 22. For example:
~~~~
K>>J = [eye(19);
    zeros(3, 8) eye(3) zeros(3, 8)];
K>>v = [1:19]';
K>>display((J * v)');
ans =

  Columns 1 through 12

     1     2     3     4     5     6     7     8     9    10    11    12

  Columns 13 through 22

    13    14    15    16    17    18    19     9    10    11

~~~~
Note that the number of condition encoded in `u.tt` should be the same
as the number of conditions *M*.

## Model fitting / inference
The toolbox includes a variety of methods to fit models to experimental
data based on the Metropolis-Hastings algorithm. This is a generic method
to sample from a target distribution (usually the distribution of the 
model parameters conditioned on experimental data). The results are therefore
an array of samples from the target distribution, which can be used to 
compute summary statistics (mean, variance) of parameters estimates.

There are currently four methods to fit models:

| Name | Hier./single subject | Description | Function |
|:----:|:-------------------:|-------------|----------|
|Flat  | Single subject     | Fits a single subject at a time. | `tapas_sem_flat_estimate.m` |
|Hier.    | Hierarchical     | Uses the population mean as prior of the parameters. | `tapas_sem_hier_estimate.m` |
|Multiv.  | Hierarchical    | Uses a linear model to construct a parametric prior from the population. | `tapas_sem_multiv_estimate.m` |
|Mixed    | Hierarchical     | Uses a mixed effect model to construct a parametric prior from the population. | `tapas_sem_mixed_estimate.m` |

Below this methods are explain in some detail.

### Single subject inference (tapas_sem_flat_estimate)
In the most simple case, the data from a subject is fitted using a standard
prior. Several conditions can be coded in `data.u.tt` and constraints 
across conditions can be implemented using a projection matrix as explained
above. An example can be found in 
`tapas/sem/examples/tapas_sem_flat_example_estimate.m`. Below we have
commented an abbreviated form of the code.

```matlab
% This function loads the data and prepares it in the necessary format.
% The data contains two conditions (pro- and antisaccade trials)
[data] = prepare_data();

% Initilize the parameters and prior of the model.
ptheta = tapas_sem_seria_ptheta(); 

% Select the likelihood function
ptheta.llh = @c_seria_multi_mixedgamma;

% Default values for the Metropolis-Hastings algorithm 
htheta = tapas_sem_seria_htheta();

% Constraints for the inversion. As in the example above, we enforce the
% constraint that parameters 9 to 11 are equal across the two conditions
ptheta.jm = [...
    eye(19)
    zeros(3, 8) eye(3) zeros(3, 8)];

pars = struct();

% This implements a multi-chain approach. It can be used to compute the
% model evidence and improve the efficiency of the algorithm. If only
% the posterior is desired use pars.T = 1;
nchains = 16;
pars.T = linspace(0.1, 1, nchains).^5;
% Number of burn-in samples
pars.nburnin = 4000;
% Number of kept samples
pars.niter = 4000;
% Number of samples between diagnostics
pars.ndiag = 1000;
% Number of times that a swap between chains is performed
pars.mc3it = 16;
% Verbosity of the diagnostics
pars.verbose = 1;

display(ptheta);
tic
% Estimate the model.
posterior = tapas_sem_flat_estimate(data, ptheta, htheta, pars);
toc

display(posterior)
``` 

The variable `ptheta` represent the parameters of the model. It is a 
structure with several fields explain in the table below.

| Field | Example value | Explanation |
|:-----:|:-------------:|:-----------:|
|mu|[11x1 double]| Prior mean of the parameters. |
|pm|[11x1 double]| Prior precision (inverse variance) of the parameters |
|p0|[11x1 double]| Expansion point (initilisation) of the algorithm |
|bdist|11| Not used |
|jm|[22x19 double]| Constraint matrix. |
|name|'seria'| Name of the model |
|llh|@c_seria_multi_mixedgamma| Likelihood function |
|lpp|@tapas_sem_prosa_lpp| Prior distribution. Shared between the PROSA and 
SERIA models |
|prepare|@tapas_sem_prepare_gaussian_ptheta| Initilisation
function of the parameters.|
|sample_priors|@tapas_sem_sample_gaussian_uniform_priors| Methods to sample
the parameters  |
|ndims|11| Number of parameters (11 for SERIA, 9 for PROSA) |

### Hierarchical inference (tapas_sem_hier_estimate)

### Parametric hierarchical inference (tapas_sem_multiv_estimate)

### Parametric mixed effect inference (tapas_sem_mixed_estimate)

# Installation

## As a python package

This toolbox can be installed as python package. Although no inference
algorithm is currently implemented, it can be potentially used in combination
with packages implementing maximum likelihood estimators or the 
Metropolis-Hasting algorithm. After installation it can be imported as
~~~~
from tapas.sem.antisaccades import likelihoods as seria
~~~~
This contains all the models described in the original
[SERIA paper](https://doi.org/10.1371/journal.pcbi.1005692).

# Installation

## Supported platforms

Mac OSX and linux are supported. We have tested in a variaty of setups
and it has worked so far. If you have any issue please contact us.

We do not support Windows but most likely it can be installed as a python 
package.

## Dependencies

* gsl/1.16>

In Ubuntu, it can be install as 
~~~~
sudo apt-get install libgsl0-dev
~~~~
To install in Mac
~~~~
brew install gsl
brew install clang-omp 
~~~~
Or alternatively using mac ports.
~~~~
sudo port install gsl
~~~~

## Matlab package

You will need a running matlab 
installation. In particular, the command line command  `matlab` should be able
to trigger matlab. The reason is that matlab is used to find out the 
matlabroot directory during the configuration. Make sure
that matlab can be triggered from the command line AND that it is not an
alias.

### Linux
To install the package it should be enough to go to
~~~~
tapas/sem/src/
~~~~
and type
~~~~
./configure && make
~~~~
The most likely problems you could face are the following:

#### Something with automake or aclocal.
In that case please install automake,f.e.,
~~~~
sudo apt-get install automake
~~~~
Then type
~~~~
autoreconf -ifv
~~~~
Then try again
~~~~
configure && make
~~~~

### Mac

This follows the same process than linux.

Most likely config will fail for one of the following reasons.

#### Has config found gls's header? 

Often after installation, the compiler fails to find gsl's headeers.
~~~~
export C_INCLUDE_PATH="$C_INCLUDE_PATH:/opt/local/include"
export CFLAGS="-I:/opt/local/include $CFLAGS"
configure && make
~~~~

#### Has config found gls's libraries? 

If not type
~~~~
export LDFLAGS="$LDFLAGS -L/opt/local/lib/ -L/usr/local/lib"
configure && make
~~~~
#### Has config found matlab?
If not, find the path of matlab and type
~~~~
export PATH=$PATH:your-matlab-path
configure && make
~~~~

## Python Package

This toolbox can be install as an usual python package using
~~~~
sudo python setup.py install 
~~~~
If you lack sudo rights or prefer not install it this way use
~~~~
python setup.py install --user
~~~~
Requirements can be installed using
~~~~
pip install -r requirements.txt
~~~~
