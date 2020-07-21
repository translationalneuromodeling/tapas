function r = tapas_sampleModel(inputs, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Samples prior values from the prior provided in the configurations of the perceptual and
% (optionally) the response model and returns the implied belief trajectories (and simulated
% responses, if a response model was specified).
%
% USAGE:
%     est = tapas_sampleModel(inputs, perceptual_config, observational_config, seed)
% 
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Store responses, inputs, and information about irregular trials in newly
% initialized structure r
%
r = dataPrep(inputs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% THE DEFAULTS DEFINED HERE WILL BE OVERWRITTEN BY ANY ARGUMENTS GIVEN WHEN CALLING tapas_fitModel.m
%
% Default perceptual model
% ~~~~~~~~~~~~~~~~~~~~~~~~
r.c_prc = tapas_ehgf_binary_config;

% Default observation model
% ~~~~~~~~~~~~~~~~~~~~~~~~~
r.c_obs = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Override default settings with arguments from the command line
if nargin > 1 && ~isempty(varargin{1})
    if isstr(varargin{1})
        r.c_prc = eval(varargin{1});
    else
        r.c_prc = varargin{1};
        % Ensure consistency of configuration of priors
        r.c_prc = tapas_align_priors(r.c_prc);
    end
end

if nargin > 2 && ~isempty(varargin{2})
    if isstr(varargin{2})
        r.c_obs = eval(varargin{2});
    else
        r.c_obs = varargin{2};
        % Ensure consistency of configuration of priors
        r.c_obs = tapas_align_priors(r.c_obs);
    end
end

% Replace placeholders in parameter vectors with their calculated values
r.c_prc.priormus(r.c_prc.priormus==99991) = r.plh.p99991;
r.c_prc.priorsas(r.c_prc.priorsas==99991) = r.plh.p99991;

r.c_prc.priormus(r.c_prc.priormus==99992) = r.plh.p99992;
r.c_prc.priorsas(r.c_prc.priorsas==99992) = r.plh.p99992;

r.c_prc.priormus(r.c_prc.priormus==99993) = r.plh.p99993;
r.c_prc.priorsas(r.c_prc.priorsas==99993) = r.plh.p99993;

r.c_prc.priormus(r.c_prc.priormus==-99993) = -r.plh.p99993;
r.c_prc.priorsas(r.c_prc.priorsas==-99993) = -r.plh.p99993;

r.c_prc.priormus(r.c_prc.priormus==99994) = r.plh.p99994;
r.c_prc.priorsas(r.c_prc.priorsas==99994) = r.plh.p99994;

r = rmfield(r, 'plh');

% Store model name also in c_sim field
r.c_sim.prc_model = r.c_prc.model;

% Set seed for random number generator
r.c_sim.seed = NaN;
if nargin > 3
    r.c_sim.seed = varargin{3};
end
    
% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Add random values to prior means, drawn from a Gaussian with prior sd
np_prc = length(r.c_prc.priorsas);
ptrans = r.c_prc.priormus + randn(1,np_prc).*sqrt(r.c_prc.priorsas);

% Transform parameters to their native space
[dummy, r.p_prc] = r.c_prc.transp_prc_fun(r, ptrans);
r.p_prc.p        = r.c_prc.transp_prc_fun(r, ptrans);
r.p_prc.ptrans   = ptrans;

% Compute perceptual states
[r.traj, infStates] = r.c_prc.prc_fun(r, r.p_prc.p);

if nargin > 2
    % Store model name also in c_sim field
    r.c_sim.obs_model = r.c_obs.model;

    % Add random values to prior means, drawn from a Gaussian with prior sd
    np_obs = length(r.c_obs.priorsas);
    ptrans = r.c_obs.priormus + randn(1,np_obs).*sqrt(r.c_obs.priorsas);

    % Transform parameters to their native space
    [dummy, r.p_obs]   = r.c_obs.transp_obs_fun(r, ptrans);
    r.p_obs.p          = r.c_obs.transp_obs_fun(r, ptrans);
    r.p_obs.ptrans     = ptrans;

    % Get function handle to observation model
    obs_fun = str2func([r.c_obs.model, '_sim']);
    
    % Simulate decisions
    r.y = obs_fun(r, infStates, r.p_obs.p);
end

end % function tapas_sampleModel

% --------------------------------------------------------------------------------------------------
function r = dataPrep(inputs)

% Initialize data structure to be returned
r = struct;

% Check if inputs look like column vectors
if size(inputs,1) <= size(inputs,2)
    disp(' ')
    disp('Warning: ensure that input sequences are COLUMN vectors.')
end

% Store inputs
r.u  = inputs;

% Determine ignored trials
ign = [];
for k = 1:size(r.u,1)
    if isnan(r.u(k,1))
        ign = [ign, k];
    end
end

r.ign = ign;

if isempty(ign)
    ignout = 'none';
else
    ignout = ign;
end
disp(['Ignored trials: ', num2str(ignout)])

% Calculate placeholder values for configuration files

% First input
% Usually a good choice for the prior mean of mu_1
r.plh.p99991 = r.u(1,1);

% Variance of first 20 inputs
% Usually a good choice for the prior variance of mu_1
if length(r.u(:,1)) > 20
    r.plh.p99992 = var(r.u(1:20,1),1);
else
    r.plh.p99992 = var(r.u(:,1),1);
end

% Log-variance of first 20 inputs
% Usually a good choice for the prior means of log(sa_1) and alpha
if length(r.u(:,1)) > 20
    r.plh.p99993 = log(var(r.u(1:20,1),1));
else
    r.plh.p99993 = log(var(r.u(:,1),1));
end

% Log-variance of first 20 inputs minus two
% Usually a good choice for the prior mean of omega_1
if length(r.u(:,1)) > 20
    r.plh.p99994 = log(var(r.u(1:20,1),1))-2;
else
    r.plh.p99994 = log(var(r.u(:,1),1))-2;
end

end % function dataPrep
