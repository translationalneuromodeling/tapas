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


## Parameters coding


### A note on the PROSA model

## Data coding
The data entered to the model is encoded as a structure with the fields
`y` and `u`. This is an structure array, in which the number of rows 
corresponds to the number of subjects. Prosaccades are encoded
as 0, and antisaccade as 0.
 
The field `y` represents the responses of subjects in terms of RT (in 
tenths of a second) and the corresponding action. 

| Fields *y* | Meaning | Data |
|:----------:|:-------:|:----:|
| *t*        | Reaction time | Tenths of second |
| *a*        | Action        | Pro=0, Anti=1 |

The field `u` represents experimental conditions. The field `tt` represents
different conditions. The coding should go from 0 to the number of 
conditions.

| Fields *u* | Meaning | Data |
|:----------:|:-------:|:----:|
|*tt*       | Trial type <br> (condition) | Integer from 0 to N |

For example, if in an experiment pro- and antisaccade trials are mixed in
a single block, it is possible to code these two types of trial as 0 and 1.
Note that the consequences of coding two types of trials as two conditions
is that a set of parameters will be initialize for each condition.

## Model fitting / inference
The toolbox includes a variety of approaches to fit models to experimental
data based on the Metropolis-Hastings algorithm. This is a generic method
to sample from a target distribution (usually the distribution of the 
model parameters conditioned on experimental data). The results are therefore
an array of samples from the target distribution, which can be used to 
compute summary statistics (mean, variance).

There are currently four methods to fit models

| Name | Hier./single subject | Description | Function |
|:----:|:-------------------:|-------------|----------|
|Flat  | Single subject     | | `tapas_sem_flat_estimate.m` |
|Hier.    | Hierarchical     | | `tapas_sem_hier_estimate.m` |
|Multiv.  | Hierarchical    | | `tapas_sem_multiv_estimate.m` |
|Mixed    | Hierarchical     | | `tapas_sem_mixed_estimate.m` |

### Single subject inference (tapas_sem_flat_estimate)

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
