function [itau] = tapas_ceode_compute_itau(P, M, U)
% [itau] = tapas_ceode_compute_itau(P, M, U)
%
% Computation of delays (in samples).
%
%   iD = delay matrix / sampling Rate
%
% The resulting sample-delay matrix is read FROM (column) TO (row)
%
%
% INPUT
%   P           struct          parameter structure
%   M           struct          model specification
%   U           struct          design specification
%
% OUTPUT
%   itau          matrix          delays in sample space
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


% Get the scaling factors for delays
if isfield(M, 'pF')
    D = M.pF.D;
else
    % Default scaling values
    switch M.f
        case 'tapas_ceode_fx_erp'
            D = [2, 16];
        case 'tapas_ceode_fx_cmc'
            D = [1, 8];
    end
end

% Get number of regions sampling rate
n = size(M.x, 1);
dt = U.dt;

% Construct delay matrix for extrinsic (De) and intrinsic (Di) delays
% (also see spm_fx_erp.m)
De = (ones(n, n) - eye(n)) .* D(2).*exp(P.D)/1000;
Di = eye(n) .* D(1) ./1000;
D = De + Di;

% Compute delays in samples
itau = D ./ dt;

end
