HGF Toolbox - Release ID: 43168dd  (HEAD, v3.0, googledrive/master, brutus/master, master)

************************************************************************
Copyright (C) 2012-2013 Christoph Mathys <chmathys@ethz.ch>
Translational Neuromodeling Unit (TNU)
University of Zurich and ETH Zurich
------------------------------------------------------------------------

The HGF toolbox is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program (see the file COPYING).  If not, see
<http://www.gnu.org/licenses/>.
************************************************************************


INTRODUCTION

The HGF toolbox provides methods for fitting time series models using
Bayesian inference.

The toolbox is provided as is. No maintenance or support is provided
or implied beyond the documentation contained in this release.

This toolbox is built around the HGF. It may therefore not contain
your favorite time series model. However, the toolbox's modular nature
should make it easy to add new models (see ADDING MODELS below).

The HGF was introduced in

Mathys C, Daunizeau J, Friston, KJ, and Stephan KE. (2011). A Bayesian
   foundation for individual learning under uncertainty. Frontiers in
   Human Neuroscience, 5:39.

Whenever you make use of one of the various models based on the HGF,
please cite this paper.

This toolbox assumes a framework where an agent in the broadest sense
(e.g., a human being, an animal, a machine, the stock market, etc.)
receives a time series of inputs to which it reacts by emitting a time
series of responses. In particular, this process is modeled by the
combination of a perceptual (sc. state space) and an observation
(sc. decision) model. The perceptual model is the time series model on
which the agent bases its responses; the observation model describes
how the agent makes decisions based on its perceptual inference.

Note that what we refer to here as the observation model describes a
"second-order" observation in the sense that the perceptual model
already contains a ("first-order") observation part that describes how
perceptual states relate to inputs. This implements the "observing the
observer" framework described in Daunizeau et al. (2010), PLoS ONE,
5(12), e15554.


USAGE

There are two ways to use the HGF toolbox:

(1) To fit various combinations of perceptual and observation models
    to observed responses:

    +---------------+   +--------------+
    |  	Observed    |   |   Stimuli    |
    |     data      | + |              |
    |  (responses)  |   |   (inputs)   |
    +---------------+   +--------------+

                    GIVEN

    +---------------+   +-------------+
    |   Perceptual  |   | Observation |
    | (state space) | + |  (decision) |
    |     model     |   |    model    |
    +---------------+   +-------------+

                     ||
                     ||
                     \/

       +------------+   +--------------+
       | Parameter  |   | Evolution of |
       |            | + |  perceptual  |
       | Estimates  |   |   states     |
       +------------+   +--------------+


(2) To simulate the trajectories of perceptual states, and responses:

    +---------------+   +--------------+
    |   Choice of   |   |   Stimuli    |
    |   parameter   | + |              |
    |    values     |   |   (Inputs)   |
    +---------------+   +--------------+

                    GIVEN

    +---------------+   +-------------+
    |   Perceptual  |   | Observation |
    | (state space) | + |  (decision) |
    |     model     |   |    model    |
    +---------------+   +-------------+

                     ||
                     ||
                     \/

    +--------------+     +-----------+
    | Evolution of |     | Simulated |
    |  perceptual  |  +  |           |
    |   states     |     | responses |
    +--------------+     +-----------+


   In simpler cases (e.g., when simply filtering inputs), only the
   evolution of the perceptual inference is of interest. The
   specification of an observation model may then simply be omitted:

    +---------------+   +--------------+
    |   Choice of   |   |   Stimuli    |
    |   parameter   | + |              |
    |    values     |   |   (Inputs)   |
    +---------------+   +--------------+

                    GIVEN

              +---------------+
              |   Perceptual  |
              | (state space) |
              |     model     |
              +---------------+

                     ||
                     ||
                     \/

              +--------------+
              | Evolution of |
              |  perceptual  |
              |   states     |
              +--------------+


MAIN FUNCTIONS

Each of the two usages has its main function. The function

    tapas_fitModel(...)

fits models to observed responses, while the function

    tapas_simModel(...)

simulates the trajectories of perceptual states, and responses. The
documentation to these functions is located at the top of their
respcective files tapas_fitModel.m and tapas_simModel.m.


INSTALLATION

Move the contents of this folder to a location of your choice.


DOCUMENTATION AND CONFIGURATION

Start Matlab, open the files tapas_fitModel.m or tapas_simModel.m, and
read the documentation there. This will point you to the relevant
configuration files.


EXAMPLES

As a simple example, start Matlab and load the example binary inputs
provided in the file example_binary_input.txt:

>> u = load('example_binary_input.txt');

First, find the Bayes optimal perceptual parameters for this dataset
under the binary HGF model:

>> bopars = tapas_fitModel([], u, 'tapas_hgf_binary_config', 'tapas_bayes_optimal_binary_config', 'tapas_quasinewton_optim_config');

You can now use the optimal parameters as prior means by adapting
tapas_hgf_binary_config.m.

Next, simulate a non-optimal agent's responses:

>> sim = tapas_simModel(u, 'tapas_hgf_binary', [NaN 0 1 NaN 1 1 NaN 0 0 NaN 1 NaN -2.5 0.01], 'tapas_unitsq_sgm', 5);
>> tapas_hgf_binary_plotTraj(sim)

The general meaning of the arguments to tapas_simModel is explained in
tapas_simModel.m. The specific meaning of each argument in this
example is explained in the configuration files of the perceptual
model (tapas_hgf_binary_config.m) and of the response model
(tapas_unitsq_sgm_config.m).

Then, try to recover these parameters by fitting the corresponding
models to the simulated data:

>> est = tapas_fitModel(sim.y, sim.u, 'tapas_hgf_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
>> tapas_fit_plotCorr(est)
>> tapas_hgf_binary_plotTraj(est)

You can also try to fit the same data using a different perceptual model:

>> est2 = tapas_fitModel(sim.y, sim.u, 'tapas_rw_binary_config', 'tapas_unitsq_sgm_config', 'tapas_quasinewton_optim_config');
>> tapas_fit_plotCorr(est2)
>> tapas_rw_binary_plotTraj(est2)

The same procedure can be applied to continuous data. The file
example_usdchf.txt contains the value of the US dollar in Swiss francs
throughout much of 2010 and 2011 (source: http://www.oanda.com).

>> usdchf = load('example_usdchf.txt');
>> bopars2 = tapas_fitModel([], usdchf, 'tapas_hgf_config', 'tapas_bayes_optimal_config', 'tapas_quasinewton_optim_config');
>> tapas_fit_plotCorr(bopars2)
>> tapas_hgf_plotTraj(bopars2)
>> sim2 = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -13  0.1 0.0001], 'tapas_gaussian_obs', 0.00005);
>> tapas_hgf_plotTraj(sim2)
>> sim3 = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 1 0.0001 0.1 0.1 0 0 0 1 1 -13  -2 0.1 0.0001], 'tapas_gaussian_obs', 0.00005);
>> tapas_hgf_plotTraj(sim3)
>> est3 = tapas_fitModel(sim2.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
>> tapas_fit_plotCorr(est3)
>> tapas_hgf_plotTraj(est3)

It is often useful to average parameters from several estimations, for
instance to compare groups of subjects. This can be achieved by using
the function tapas_bayesian_parameter_average(...) which takes into
account the covariance structure between the parameters and weights
individual estimates according to their precision:

>> sim4 = tapas_simModel(usdchf, 'tapas_hgf', [1.04 1 0.0001 0.1 0 0 1 -15  0.1 0.0001], 'tapas_gaussian_obs', 0.00005);
>> tapas_hgf_plotTraj(sim4)
>> est4 = tapas_fitModel(sim4.y, usdchf, 'tapas_hgf_config', 'tapas_gaussian_obs_config', 'tapas_quasinewton_optim_config');
>> tapas_fit_plotCorr(est4)
>> tapas_hgf_plotTraj(est4)
>> bpa = tapas_bayesian_parameter_average(est3, est4);
>> tapas_fit_plotCorr(bpa)
>> tapas_hgf_plotTraj(bpa)

Note that Bayesian parameter averaging only works for estimates that
are based on the same priors and should only be used with care for
estimates based on different inputs.


DEALING WITH NEW DATASETS

When beginning to analyze a new dataset, it is important to make a
sensible choice of priors. The toolbox helps with this in three ways:

(1) The function tapas_fitModel can calculate the Bayes optimal
    parameter values given the inputs, a perceptual and a response
    model, and their respective priors (see the documentation in
    tapas_fitModel.m). These can then be used as prior means in the
    further analysis.

(2) For continuous inputs, the configuration files of the perceptual
    models accept placeholders that are replaced by values derived
    from the inputs at runtime (see the documentation in the relevant
    configuration files). This makes it easy to automatically set, for
    example, the prior mean of the main quantity of interest the value
    of the first input.

(3) For the HGF with continuous inputs, the upper bound on theta can
    be automatically set to the highest value for which the
    assumptions underlying the variational inversion of the HGF still
    hold. See tapas_hgf_config.m for details.


ADDING MODELS

The modularity of this toolbox enables you to add perceptual and
observation models of your choice. This requires the following
functions that tapas_fitModel(...) and tapas_simModel(...) will expect
to find (replace <modelname> by the name of your model):

tapas_<modelname>            contains the model machinery
tapas_<modelname>_config     contains the configuration settings (only for
			     tapas_fitModel(...))
tapas_<modelname>_transp     transforms parameters from the space they are
                             estimated in to their native space (only for
		             tapas_fitModel(...) and
			     tapas_bayesian_parameter_average(...))
tapas_<modelname>_namep      returns a structure of named parameters (only
                             for tapas_simModel(...))

Additionally, for observation models, tapas_simModel(...) expects to
find a function that performs the simulation of responses:

tapas_<modelname>_sim

For details, look at the corresponding files of an existing model
(e.g. tapas_hgf_binary) and use them as templates.


ADDING OPTIMIZATION ALGORITHMS

To add a new optimization algorithm, provide the following functions
that tapas_fitModel(...) will expect to find (replace <algo> by the
name of your algorithm):

tapas_<algo>           contains the machinery of your algorithm
tapas_<algo>_config    contains the configuration settings

For details, look at the corresponding files of an existing algorithm
(e.g. tapas_quasinewton_optim) and use them as templates.
