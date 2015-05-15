function test_tapas_mpdcm_fmri_map(fp)
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

for i = 1:5
    [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(d(i));

    % Test whether there is any clear bug
    try
        tapas_mpdcm_fmri_map(y, u, theta, ptheta);
        fprintf(fp, '       Passed\n');
    catch err
        fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
        display(getReport(err, 'extended'));
    end
end

try
    [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(d);
    tapas_mpdcm_fmri_map(y, u, theta, ptheta);
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    display(getReport(err, 'extended'));
end


end

