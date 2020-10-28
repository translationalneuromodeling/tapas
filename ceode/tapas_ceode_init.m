function [  ] = tapas_ceode_init()
% [  ] = tapas_ceode_init()
%
% Intitialization function of the tapas/ceode toolbox. It adds the relevant
% toolbox function to the path. Additionally, it tests for the availability
% of an SPM version, and performs a simple test to investigate
% compatibility between the toolboxes used.
%
% INPUT
%
% OUTPUT
%
% -------------------------------------------------------------------------
%
% Author: Dario Schöbi
% Created: 2020-08-10
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS ceode Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------


% Adds tapas/ceode/code folder to the current path
ceode_root = fileparts(which('tapas_ceode_init'));
addpath(genpath(fullfile(ceode_root, 'code')));


%------------------------- Check for SPM ----------------------------------
fprintf(['\n---------------------------------------------------------' ...
    '\n Checking for SPM version \n' ...
    '--------------------------------------------------------- \n'])
try
    spm_version = spm('version');
    fprintf(['You are currently using SPM version: ' spm_version]);
catch
    spm_version = [];
    error(['Please add the SPM EEG toolboxes to your path']);
end


%-------------------------- Test Scripts ----------------------------------
fprintf(['\n\n---------------------------------------------------------' ...
    '\n Running test scripts \n' ...
    '--------------------------------------------------------- \n'])

try
    erp_test = tapas_ceode_testScript_erp();
catch
    erp_test.score = 0;
    erp_test.msg = sprintf(['ERP test unsuccessful: \n' ...
        'Please make sure that the SPM EEG toolboxes are in your ' ...
        'MATLAB path.\n' ...
        'Please make sure that the test scripts have not been changed.']);
end

try
    cmc_test = tapas_ceode_testScript_cmc();
catch
    cmc_test.score = 0;
    cmc_test.msg = sprintf(['CMC test unsuccessful: \n' ...
        'Please make sure that the SPM EEG toolboxes are in your ' ...
        'MATLAB path.\n' ...
        'Please make sure that the test scripts have not been changed.']);
end

% Check the Test results
if erp_test.score
    fprintf('%s \n', erp_test.msg);
else
    warning('%s \n', erp_test.msg);
end
if cmc_test.score
    fprintf('%s', cmc_test.msg);
else
    warning('%s', cmc_test.msg);
end


%------------------------ Current Integrator Specs ------------------------
if spm_version
    fprintf(['\n\n---------------------------------------------------------' ...
        '\n Checking integrator settings \n' ...
        '--------------------------------------------------------- \n'])
    
    [fx_erp, fx_cmc, int_spec] = check_integrator_specification();
    fprintf(['The current integration runs with: ' ...
        '\n Integrator: %s' ...
        '\n Dynamical Equations ERP: %s' ...
        '\n Dynamical Equations CMC: %s \n'], ...
        int_spec, fx_erp, fx_cmc);
    
    check_integrator_compatibility(fx_erp, fx_cmc, int_spec)
    
end


end


function [fx_erp, fx_cmc, int_spec] = ...
    check_integrator_specification()
% Checks the current dynamical equation function and the integrator used

% Create a dummy parameter
P.A{1} = zeros(2, 2);
[~, fx_erp] = spm_dcm_x_neural(P, 'erp');
[~, fx_cmc] = spm_dcm_x_neural(P, 'cmc');

% Get the dependencies of spm_gen_erp
[fList] = matlab.codetools.requiredFilesAndProducts('spm_gen_erp.m', ...
    'toponly');

% Find the integrator
int_spec = '';
for i = 1 : length(fList)
    [~, tmp] = fileparts(fList{i});
    if strcmp(tmp, 'spm_int_L')
        int_spec = tmp;
    elseif strcmp(tmp, 'tapas_ceode_int_euler')
        int_spec = tmp;
    end
end

end


function check_integrator_compatibility(fx_erp, fx_cmc, int_spec)
% Check the combination of the dynamical equations and the integrator for
% compatiblity

if ~strcmp(int_spec, 'tapas_ceode_int_euler')
    error(sprintf(['\nYour setup currently does not run the tapas/ceode integrator.' ...
        '\nPlease consult the README.md how to enable ceode.\n']));
else
    fprintf('\nYour setup currently runs with the tapas/ceode integrator.\n');
end

if ~strcmp(fx_erp, 'tapas_ceode_fx_erp')
    warning('Dynamical Equations: ERP are not compatible with integrator');
end

if ~strcmp(fx_cmc, 'tapas_ceode_fx_cmc')
    warning('Dynamical Equations: CMC are not compatible with integrator');
end

fprintf('\nPlease consult the README.md if you want to change the settings.\n');


end


