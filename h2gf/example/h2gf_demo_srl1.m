% h2gf demo using the data: PRSSI, EEG, short version of SRL (srl1)
%
% missed trials are removed and therefore not part of the perceptual model
% =========================================================================
% h2gf_demo_srl1(1,4000,1,1)
% =========================================================================

function h2gf_demo_srl1(m,NrIter,spec_eta,config_file)

addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));

%% We load the behavioural srl1 data
data_srl1 = tapas_h2gf_load_example_data_srl1();
% Number of subjects
num_subjects = length(data_srl1);

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
    eta_v = [1 1 1 1 1 1 1 1 1 1 1 1 1 5 1]';
    % mu1_0, mu2_0, mu3_0, sa1_0, sa2_0, sa3_0,
    % rho1, rho2, rho3, ka1, ka2, om1, om2, om3
    % ze
elseif spec_eta == 6
    eta_v = [1 1 1 1 1 1 1 1 1 1 1 1 1 10 1]';
    % mu1_0, mu2_0, mu3_0, sa1_0, sa2_0, sa3_0,
    % rho1, rho2, rho3, ka1, ka2, om1, om2, om3
    % ze
end

%% Prepare the model
% Initialize a structure to hold the hgf
hgf = struct('c_prc', [], 'c_obs', []);
% Set up the number of levels
hgf.c_prc.n_levels = 3;
% Set up the perceptual function
hgf.c_prc.prc_fun = @tapas_hgf_binary;

% Set up the reparametrization function
hgf.c_prc.transp_prc_fun = @tapas_hgf_binary_transp;

% Set up the observation function.
hgf.c_obs.obs_fun = @tapas_unitsq_sgm;
% Reparametrization function
hgf.c_obs.transp_obs_fun = @tapas_unitsq_sgm_transp;

% Enter the configuration of the binary hgf
if config_file == 1
    config = tapas_hgf_binary_config_estka2_new();
    configtype = 'estka2';
elseif config_file == 2
    config = tapas_hgf_binary_config_estka2mu2_new();
    configtype = 'estka2mu2';
elseif config_file == 3
    config = tapas_hgf_binary_config_estka2mu3_new();
    configtype = 'estka2mu3';
elseif config_file == 4
    config = tapas_hgf_binary_config_estka2om3_new();
    configtype = 'estka2om3';
elseif config_file == 5
    config = tapas_hgf_binary_config_estka2sa2_new();
    configtype = 'estka2sa2';
elseif config_file == 6
    config = tapas_hgf_binary_config_estka2sa3_new();
    configtype = 'estka2sa3';
elseif config_file == 7
    config = tapas_hgf_binary_config_estom2_new();
    configtype = 'estom2';
elseif config_file == 8
    config = tapas_hgf_binary_config_estom2mu2_new();
    configtype = 'estom2mu2';
elseif config_file == 9
    config = tapas_hgf_binary_config_estom2mu3_new();
    configtype = 'estom2mu3';
elseif config_file == 10
    config = tapas_hgf_binary_config_estom2om3_new();
    configtype = 'estom2om3';
elseif config_file == 11
    config = tapas_hgf_binary_config_estom2sa2_new();
    configtype = 'estom2sa2';
elseif config_file == 12
    config = tapas_hgf_binary_config_estom2sa3_new();
    configtype = 'estom2sa3';
end

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
h2gf_est_srl1 = tapas_h2gf_estimate(data_srl1, hgf, inference, pars);

display(h2gf_est_srl1);
cd (maskResFolder);
save(['h2gf_3l_est_srl1_',configtype,'_eta',eta_label,'_', num2str(NrIter),'_',num2str(m),'.mat'],'-struct','h2gf_est_srl1');
clear h2gf_est_srl1;
end