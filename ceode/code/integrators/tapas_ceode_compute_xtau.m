function [xx] = tapas_ceode_compute_xtau(tn, xhist, itau)
% [xx] = tapas_ceode_compute_xtau(P, M, U)
%
% Computation of delayed states according to the continuous extension for
% ODE methods. Delayed states are computed as a linear interpolation
% between sampled time points
%
%   xhat(t-tau) = x_{i-1} + (t-tau)/dt * (x_{i}-x_{i-1})
%
%
% INPUT
%   tn           int            current sample of integration
%   xhist        xhist          history of states
%   itau         itau           delays in sample space
%
% OUTPUT
%   xx           cell           delayed states. Cell read as from (columns)
%                               to (rows)
%
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


% Initialize auxiliary variables and allocate space
xt = cell(size(itau));
nStates = size(xhist, 1);
nSources = size(itau, 1);
nx = nStates / nSources;

% Iterate over all delays, and compute the linear interpolation to the
% activity
for i = 1 : size(itau, 1)
    for j = 1 : size(itau, 2)
        
        lxt = zeros(size(xhist, 1), 1);
        uxt = zeros(size(xhist, 1), 1);
        
        % Samples immediately before and after the delayed state
        uitau = ceil(itau);
        litau = floor(itau);
        
        % Normalize delay to the interval between samples
        tp = mod(itau, 1);
        
        % If current time < delays, initialize to 0 activity (initial
        % conditions)
        if tn - uitau(j, i) <= 0
            uxt = zeros(nStates, 1);
            lxt = zeros(nStates, 1);
        else
            lxt = xhist(:, tn - uitau(j, i));
            uxt = xhist(:, tn -litau(j, i));
        end
        
        % Interpolation step
        xt{j, i} = spm_unvec(lxt + tp(j, i) .* (uxt - lxt), ...
            zeros(nSources, nx));
        
    end    
end

% Reshape the delayed states into a compressed format
for i = 1 : nSources
    for j = 1 : nSources
        xx{i}(j, :) = xt{i, j}(j, :);
    end
end

end
