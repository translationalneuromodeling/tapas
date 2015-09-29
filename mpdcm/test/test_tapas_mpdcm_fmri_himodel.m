function test_tapas_mpdcm_fmri_himodel(fp)
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


dcm = cell(10, 1);
%dcm = test_tapas_mpdcm_fmri_load_td();
odcm = mvapp_load_dcms();
dcm(:) = odcm(6);

rng(1987)

for i = 1:10
    dcm{i}.Y.y = dcm{i}.Y.y + 3.0 * randn(size(dcm{i}.Y.y));
end


% Test whether there is any clear bug
try
    pars = struct('niter', 10);
    ptheta.x = ones(10, 1); % Only the base offset 
    tapas_mpdcm_fmri_himodel(dcm, ptheta);
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end

% Test whether there is any clear bug
try
    pars = struct('niter', 10);
    ptheta.x = ones(10, 2); % Only the base offset 
    ptheta.x(:, 2) = 1:10;
    tapas_mpdcm_fmri_himodel(dcm, ptheta);
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end


% Test whether there is any clear bug
try
    pars = struct('niter', 10);
    ptheta.x = ones(10, 3); % Only the base offset 
    ptheta.x(:, 2) = 1:10;
    ptheta.x(:, 3) = sin(0.5*pi*(1:10));
    tapas_mpdcm_fmri_himodel(dcm, ptheta);
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    fprintf(fp, getReport(err, 'extended'));
end


end

