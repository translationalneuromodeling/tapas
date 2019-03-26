function [isConstant, dy, verbose] = tapas_physio_detect_constants(y, ...
    nMinConstantSamples, deltaMaxDiff, verbose)
% Detects constant portions of input time series, e.g. to flag breathing
% belt detachment/clipping
%
%[isConstant, dabsY, verbose] = tapas_physio_detect_constants(y, ...
%    nMinConstantSamples, deltaMaxDiff, verbose)
%
% IN
%   y   [nSamples,1] time course, e.g. breathing ons_secs.r
%   nMinConstantSamples 
%       number of subsequent samples that have to be constant to be flagged
%       as a constant portion of the time series. (default = 10)
%   deltaMaxDiff
%       maximum difference of subsequent samples to be considered equal
%       default = single precision (1.1921e-07)
%   verbose     verbosity structure see tapas_physio_new
% OUT
%   r   [nSamples,1] = 0 for all samples that are not constant, = 1 for all
%       samples that belong to a constant time window of the time course
%
% EXAMPLE
%   tapas_physio_detect_constants
%
%   See also tapas_physio_new tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2016-09-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 2
    nMinConstantSamples = 10;
end

if nargin < 3 || isempty(deltaMaxDiff)
    deltaMaxDiff = eps('single');
end

if nargin < 4
    verbose = struct('level', 4, 'fig_handles', []);
end

DEBUG = verbose.level >= 4;

dy = diff(y);

% add nMinConstantSamples - 1 to get right index in original time series
% odx = first level of indices, indices in (d)y
idxIsConstant = find(abs(dy) < deltaMaxDiff);

if nMinConstantSamples > 2
    
    % label vector for constants has to be determined via sequence lengths
    
    % train of constant values = their indices are consecutive!
    % therefore: find train sequences of consecutive numbers, as in
    % https://ch.mathworks.com/matlabcentral/answers/86420-find-a-series-of-consecutive-numbers-in-a-vector
    a=diff(idxIsConstant);
    b=find([a; inf]>1); % >1 = non-consecutive index = jump in indices = end of train
    c=diff([0; b]); % between b indices (index jumps!), there is only consecutive index changes, and we can count the 1 to get the length of the train
    d=cumsum(c); % endpoint indices of the sequences in a; since b contains index values of a, and its differences (sequence lengths) are summed up here, this works!
    
    % define start and end points of constant sequences
    idx2IsConstantEnd = d; % idx2, since it is the 2nd level of indices, the indices *within idxIsConstant*, not within (d)y
    idx2IsConstantStart = d - c + 1;
    
    % populate indices for labeling constants for all sequences (of
    % consecutive indices) with sufficient length
    % idx3, since it is the 3rd level of indices: indices of the idx2
    % vectors
    idx3ValidLength = find(c >= nMinConstantSamples - 1); % -1 because of diff!
    
    idx2IsConstantEnd = idx2IsConstantEnd(idx3ValidLength);
    idx2IsConstantStart = idx2IsConstantStart(idx3ValidLength);
    nValidSequences = numel(idx3ValidLength);
    idx2IsConstantValidLength = [];
    for s = 1:nValidSequences
        idx2IsConstantValidLength = [idx2IsConstantValidLength; ...
            (idx2IsConstantStart(s):idx2IsConstantEnd(s))'];
    end
    
    % now back to index vector in 
    idxIsConstantValidLength = idxIsConstant(idx2IsConstantValidLength);
    
else % take all indices, no selection by length of sequence of constant values!
    idxIsConstantValidLength = idxIsConstant;
end

% now we also have to add the values after the detected ones, because of
% diff, there is an index shift, and because a small diff stems from TWO
% constants
% to avoid resorting, we make [n1, n1+1; n2, n2+1; ...] 
% -> [n1, n2, ...; n1+1, n2+1, ...] -> reshape takes all values in first
% row first then: [n1; n1+1; n2; n2+1; ...)
idxIsConstantFinal = unique(reshape([idxIsConstantValidLength, idxIsConstantValidLength+1]', [],1));  

% create output binary vector
isConstant = zeros(size(y));
isConstant(idxIsConstantFinal) = 1;

if DEBUG
    fh = tapas_physio_get_default_fig_params();
    verbose.fig_handles(end+1) = fh;
    stringTitle = 'Preproc: Detection of suspicious constant values in physiological time series';
    set(fh, 'Name', stringTitle);
    plot(dy/max(dy));
    hold all;
    plot(y);
    plot(isConstant);
    ylim([min(y), max(y)]);
    title(stringTitle);
    legend('\Delta|y| (normalized)', 'y (Physiological Time Series)',  ...
       'isConstant');
end