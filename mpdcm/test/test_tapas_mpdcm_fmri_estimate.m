function test_tapas_mpdcm_fmri_estimate(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);


d = test_tapas_mpdcm_fmri_load_td();

% Check that there are no fatal error

nd = d{1};
nd.U.u = nd.U.u(1:4:end,:);
nd.U.dt = 1.0;

try
    pars = struct();
    pars.T = linspace(1e-1, 1, 100).^5;
    pars.nburnin = 1000;
    pars.niter = 1000;
    pars.mc3 = 1;
    pars.verbose = 0;
    dcm = tapas_mpdcm_fmri_estimate(nd, pars);   
    display('    Passed')
catch err
    db = dbstack();
    fprintf('   Not passed at line %d\n', db(1).line)
    disp(getReport(err, 'extended'));
end

% Make a more intensive test of the function
for i = 1:5
    try
        pars = struct();
        pars.T = linspace(1e-1, 1, 100).^5;
        pars.nburnin = 50;
        pars.niter = 100;
        pars.mc3 = 1;
        pars.verbose = 0;
        tic
        dcm = tapas_mpdcm_fmri_estimate(d{i}, pars);
        toc
        fprintf('fe: %0.5f\n', dcm.F);
        display('    Passed')
    catch err
        db = dbstack();
        fprintf('   Not passed at line %d\n', db(1).line)
        disp(getReport(err, 'extended'));
    end
end
end
