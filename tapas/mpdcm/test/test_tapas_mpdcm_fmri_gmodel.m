function test_tapas_mpdcm_fmri_gmodel(fp)
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

for i = 1:numel(d)

    [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(d(i));

    % Test whether there is any clear bug
    try
        [q, otheta] = tapas_mpdcm_fmri_gmodel(y, u, theta, ptheta);
        fprintf(fp, '       Passed\n');
    catch err
        fprintf(fp, '   Not passed at line %d\n', err.stack(1).line);
        disp(getReport(err, 'extended'));
    end
end

rng(1987);
for i = 1:numel(d)
    d{i}.Y.y = d{i}.Y.y + 3.0 * randn(size(d{i}.Y.y));
end

for i = 1:numel(d)

    [y, u, theta, ptheta] = tapas_mpdcm_fmri_tinput(d(i));

    % Test whether there is any clear bug
    try
        [q, otheta] = tapas_mpdcm_fmri_gmodel(y, u, theta, ptheta);
        fprintf(fp, '       Passed\n');
    catch err
        fprintf(fp, '   Not passed at line %d\n', err.stack(1).line);
    end
end



end

