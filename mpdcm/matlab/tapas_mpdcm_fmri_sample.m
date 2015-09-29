function [np] = tapas_mpdcm_fmri_sample(op, ptheta, htheta, v)
%% Draws a new sample from a Gaussian proposal distribution.
%
% Input
%   op -- Old parameters
%   ptheta -- Prior
%   htheta -- Hyperpriors
%   v -- Kernel. Two fields: s which is a scaling factor and S which is the     
%       Cholosvky decomposition of the kernel.
%
% Ouput
%   np -- New output 
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

if nargin < 4
    s = cell(numel(op, 1));
    s{:} = 1;
    S = cell(numel(op, 1));
    S{:} = eye(numel(op{1}));
    v = struct('S', S, 's', s);
end

nt = numel(op);
np = cell(size(op));

for i = 1:nt
    np{i} = full(op{i} + v(i).s * v(i).S *randn(size(op{i})));
end

end
