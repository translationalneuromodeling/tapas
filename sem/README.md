# README

aponteeduardo@gmail.com
copyright (C) 2015-2017

# The SERIA model

The [SERIA model](http://www.biorxiv.org/content/early/2017/06/08/109090)
is a formal statistical model of the probability of a 
pro- or antisaccade and reaction time. The currente toolbox includes an 
inference method based on the Metropolis-Hasting algorithm implemented in
MATLAB.

After installation (see below) and starting tapas, you can run an
example using

~~~~
tapas_init();
tapas_sem_example_invesion(1);
~~~~

This will load data and estimate the parameters. The data consists
of a list of trials with trial type (pro or antisaccade), the
action performed (pro or antisaccade) and the reaction time. 

You can use the file `tapas/sem/examples/tapas_sem_example_inversion`
as a template to run your analysis.

## As a python package

This toolbox can be installed as python package. Although no inference
algorithm is currently included, it can be potentially used in combination
with packages implementing
maximum likelihood estimators or the metropolis algorithm. After 
installation it can be imported as

~~~~
from tapas/sem/antisaccades import likelihoods as seria
~~~~

This contains all the models described in the SERIA paper.

# Installation

## Supported platforms

Mac and linux platforms are supported. We have tested a variaty of setups
and it has worked so far. If you have any issue please contact us.

In OSX, currently we do not support openmp as clang doesn't directly support
it. Although it is possible to use openmp it is not trivial. If you are
interested please contact us.

We do not support Windows but most likely it can be installed as a python 
package.

## Dependencies

* gsl/1.16

In Ubuntu, it can be install as 

~~~~
sudo apt-get install libgsl0-dev
~~~~

To install in mac you will need to install gsl

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

To install the package it should be enough to go to

~~~~
tapas/sem/src/
~~~~

and type

~~~~
./configure && make
~~~~

The most likely problems you could face are the following:

* Something with automake or aclocal. In that case please install automake,
f.e.,

~~~~
sudo apt-get install automake
~~~

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

Has config found gls's header? Often after installation, the compiler will
fail to find the headers. 

~~~~
export C_INCLUDE_PATH="$C_INCLUDE_PATH:/opt/local/include"
export CFLAGS="-I:/opt/local/include $CFLAGS"
configure && make
~~~~

Has config found gls's libraries? If not type

~~~~
export LDFLAGS="$LDFLAGS -L/opt/local/lib/ -L/usr/local/lib"
configure && make
~~~~

Has config found matlab? If not, find the path of matlab and type

~~~
export PATH=$PATH:your-matlab-path
configure && make
~~~

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

If you have any question please contact us.

