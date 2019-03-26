function R = tapas_physio_scaleorthmean_regressors(cardiac_sess)
% create orthogonalized cardiac regressors (Fourier expansion) to ensure
% numerical stability of SVD in spm_spm, if regressors are highly
% correlated (e.g. after triggered acquisition)
%
% USAGE:
%   R = tapas_physio_scaleorthmean_regressors(cardiac_sess)
%
% -------------------------------------------------------------------------
% INPUT:
%   cardiac_sess    - RETROICOR-generated cardiac regressors for a design
%                     matrix [Nscans X (2*expansion_order)]
%
% -------------------------------------------------------------------------
% OUTPUT:
%   cardiac_sess    - scaled to max, orthogonalized, mean-free cardiac regressors
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if isempty(cardiac_sess)
    R = [];
else
    R       = scale_max(spm_orth(mean_centre(cardiac_sess)));
end

function Rout = mean_centre( R )
Rout = R - repmat(mean(R), length(R), 1);

function Rout = scale_max( R )
Rout = R./repmat(max(abs(R)), length(R), 1);
