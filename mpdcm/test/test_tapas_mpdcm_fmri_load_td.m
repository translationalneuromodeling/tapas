function [d] = test_dcm_fmri_load_td(fp)
%% Loads test data into a cell array.
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

% Get current location
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

d = cell(5, 1);

d{1} = load(fullfile(tdir, 'data', 'd1.mat'));
d{2} = load(fullfile(tdir, 'data', 'd2.mat'));
d{3} = load(fullfile(tdir, 'data', 'd3.mat'));
d{4} = load(fullfile(tdir, 'data', 'd4.mat'));
d{5} = load(fullfile(tdir, 'data', 'd5.mat'));

d{1} = d{1}.DCM;
d{2} = d{2}.DCM;
d{3} = d{3}.DCM;
d{4} = d{4}.DCM;
d{5} = d{5}.DCM;

end

