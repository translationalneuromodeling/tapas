% h2gf demo using the data: PRSSI, EEG, short version of SRL (srl2)
%
% missed trials are removed and therefore not part of the perceptual model
% =========================================================================
% h2gf_demo_srl2_rw(1,4000,1,1)
% =========================================================================

function h2gf_demo_srl2_rw(m,NrIter,spec_eta)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

%% We load the behavioural srl2 data
data_srl2 = tapas_h2gf_load_example_data_srl2();
% Number of subjects
num_subjects = length(data_srl2);

disp(['Nr samples stored:', num2str(NrIter)]);
disp('**************************************');
disp(['eta set to:', num2str(spec_eta)]);
disp('**************************************');

%% specify eta:
eta_label = num2str(spec_eta);
if spec_eta == 1
    eta_v = spec_eta;
elseif spec_eta == 2
    eta_v = 10;
elseif spec_eta == 3
    eta_v = 20;
elseif spec_eta == 4
    eta_v = 40;
elseif spec_eta == 5
    eta_v = [1 1 5 1 1]';
    % v_0mu, v_0sa, almu, alsa, ze
elseif spec_eta == 6
    eta_v = [1 1 10 1 1]';
    % v_0mu, v_0sa, almu, alsa, ze
end

%% Prepare the model
% Initialize a structure to hold the hgf
hgf = struct('c_prc', [], 'c_obs', []);
% Set up the number of levels
hgf.c_prc.n_levels = 3;
% Set up the perceptual function
hgf.c_prc.prc_fun = @tapas_rw_binary;

% Set up the reparametrization function
hgf.c_prc.transp_prc_fun = @tapas_rw_binary_transp;

% Set up the observation function.
hgf.c_obs.obs_fun = @tapas_unitsq_sgm;
% Reparametrization function
hgf.c_obs.transp_obs_fun = @tapas_unitsq_sgm_transp;

% Enter the configuration of the binary hgf

config = tapas_rw_binary_config();
configtype = 'estrw';

disp(['config file:', configtype]);
disp('**************************************');
% Priors of the perceptual model
hgf.c_prc.priormus = config.priormus;
hgf.c_prc.priorsas = config.priorsas;

% Priors of the observational model
hgf.c_obs.priormus = log(48);
hgf.c_obs.priorsas = 1;

% Set the empirical prior
% Eta weights the prior with respect to the observations. Because the prior
% mean mu is treated as fixed observations, eta is the number of observations
% represented by mu. If eta = 1, mu is treated as a single additional observation.
hgf.empirical_priors = struct('eta', []);
% eta can be a scalar of a vector. If eta is a vector, it should have
% the dimensionality of mu.
hgf.empirical_priors.eta = eta_v;


%% Parameters for inference.
% Initialize the place holder for the parameters of the
% inference. Missing parameters are filled by default
% values. This is implemented in tapas_h2gf_inference.m

inference = struct();
pars = struct();

% Number of samples stored
pars.niter = NrIter;
% Number of samples in the burn-in phase
pars.nburnin = 1000;
% Number of samples used for diagnostics. During the
% burn-in phase the parameters of the algorithm are
% adjusted to increase the efficiency. This happens after
% every diagnostic cycle.
pars.ndiag = 100;

% Set up the so called temperature schedule. This is used to
% compute the model evidence. It is a matrix of NxM, where N
% is the number of subjects and M is the number of chains used
% to compute the model evidence. The
% temperature schedule is selected using a 5th order power rule.
pars.T = ones(num_subjects, 1) * linspace(0.01, 1, 8).^5;

% This controls how often a 'swap' step is perform.
pars.mc3it = 0;

%% define where to store results:
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

maskResFolder = ([tdir,'/results/',configtype,'/eta', eta_label,'/', num2str(NrIter)]);
mkdir(maskResFolder);
%% Run the inference method
% This function is entry point to the algorithm. Note that its
% behavior can be largely modified by changing the default
% settings.

%% run inference multiple times
h2gf_est_srl2 = tapas_h2gf_estimate(data_srl2, hgf, inference, pars);

display(h2gf_est_srl2);
cd (maskResFolder);
save(['h2gf_rw_est_srl2_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat'],'-struct','h2gf_est_srl2');
clear h2gf_est_srl2;
end