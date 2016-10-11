function [isConstant, dy] = tapas_physio_detect_constants(y, nMinConstantSamples, deltaMaxDiff)
% Detects constant portions of input time series, e.g. to flag breathing
% belt detachment/clipping
%
%   output = tapas_physio_detect_constants(input)
%
% IN
%   y   [nSamples,1] time course, e.g. breathing ons_secs.r
%   nMinConstantSamples 
%       number of subsequent samples that have to be constant to be flagged
%       as a constant portion of the time series. (default = 10)
%   deltaMaxDiff
%       maximum difference of subsequent samples to be considered equal
%       default = single precision (1.1921e-07)
% OUT
%   r   [nSamples,1] = 0 for all samples that are not constant, = 1 for all
%       samples that belong to a constant time window of the time course
%
% EXAMPLE
%   tapas_physio_detect_constants
%
%   See also
%
% Author: Lars Kasper
% Created: 2016-09-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id$
DEBUG = false;

if nargin < 2
    nMinConstantSamples = 10;
end

if nargin < 3
    deltaMaxDiff = eps('single');
end

dy = [ones(nMinConstantSamples-1,1); diff(y,nMinConstantSamples-1)];

% add nMinConstantSamples - 1 to get right index in original time series
idxIsConstant = find(abs(dy) < deltaMaxDiff);

% flag other nMinConstantSamples - 1 samples after detected ones which
% induced the detection of so many constants
idxIsConstantFilled = reshape(idxIsConstant, 1, []);
for n = 1:numel(idxIsConstant)
    idxIsConstantFilled = unique([idxIsConstantFilled, ...
        idxIsConstant(n) - (1:(nMinConstantSamples-1))]);
end


% create output binary vector
isConstant = zeros(size(y));
isConstant(idxIsConstantFilled) = 1;

if DEBUG
    figure; plot(y);hold all;
    plot(isConstant);plot(dy);
    legend('y', 'isConstant', 'dy');
end