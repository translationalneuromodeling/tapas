function [hgf] = tapas_hgf_conf_model(responses, inputs, varargin)
%% See the documentation of tapas_fitModel. 
%
% Input
%       
% Output
%       

% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
% aponteeduardo@gmail.com
% copyright (C) 2016
%

r = dataPrep(responses, inputs);
r.c_prc = tapas_hgf_binary_config;
r.c_obs = tapas_unitsq_sgm_config;
r.c_opt = tapas_quasinewton_optim_config;

if nargin > 2 && ~isempty(varargin{1})
    r.c_prc = eval(varargin{1});
end

if nargin > 3 && ~isempty(varargin{2})
    r.c_obs = eval(varargin{2});
end

if nargin > 4 && ~isempty(varargin{3})
    r.c_opt = eval(varargin{3});
end

% Replace placeholders in parameter vectors with their calculated values
r.c_prc.priormus(r.c_prc.priormus==99991) = r.plh.p99991;
r.c_prc.priorsas(r.c_prc.priorsas==99991) = r.plh.p99991;

r.c_prc.priormus(r.c_prc.priormus==99992) = r.plh.p99992;
r.c_prc.priorsas(r.c_prc.priorsas==99992) = r.plh.p99992;

r.c_prc.priormus(r.c_prc.priormus==99993) = r.plh.p99993;
r.c_prc.priorsas(r.c_prc.priorsas==99993) = r.plh.p99993;

r.c_prc.priormus(r.c_prc.priormus==99994) = r.plh.p99994;
r.c_prc.priorsas(r.c_prc.priorsas==99994) = r.plh.p99994;

r = rmfield(r, 'plh');

hgf = r;

end

% --------------------------------------------------------------------------------------------------
function r = dataPrep(responses, inputs)

% Initialize data structure to be returned
r = struct;

% Check if inputs look like column vectors
if size(inputs,1) <= size(inputs,2)
    disp(' ')
    disp('Warning: ensure that input sequences are COLUMN vectors.')
end

% Store responses and inputs
r.y  = responses;
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
    
% Determine irregular trials
irr = [];
for k = 1:size(r.y,1)
    if isnan(r.y(k,1))
        irr = [irr, k];
    end
end

% Make sure every ignored trial is also irregular
irr = unique([ign, irr]);

r.irr = irr;

if isempty(irr)
    irrout = 'none';
else
    irrout = irr;
end
disp(['Irregular trials: ', num2str(irrout)])
    
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

end



