function [ u ] = tapas_huge_boxcar( dt, nBoxes, period, onRatio, padding )
% Generate a boxcar function for use as experimental stimulus. All
% timing-related arguments must be specified in seconds. 
% 
% INPUTS:
%   dt      - Numeric scalar indicating sampling time interval.
%   nBoxes  - Vector indicating number of blocks.
%   period  - Vector containing time interval between block onsets.
%   onRatio - Vector containing ratio between block length and 'period'.
%             Must be between 0 and 1.
% 
% OPTIONAL INPUTS:
%   padding - Length of padding at the beginning and end.
% 
% OUTPUTS:
%   u - A cell array containing the boxcar functions.
% 
% EXAMPLES:
%   u = TAPAS_HUGE_BOXCAR(.01, 10, 3, 2/3, [4 0])    Generate boxcar
%       function with 10 blocks, each 2 seconds long with 1 second inter
%       block interval and onset of first block at 4 seconds.
% 
% See also tapas_Huge.SIMULATE
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


%% check input
nSignals = numel(nBoxes);

if isscalar(period) && nSignals > 1
    period = repmat(period, nSignals, 1);
else
    assert(numel(period) == nSignals, 'TAPAS:HUGE:Boxcar:InputSize', ...
        'Size of period must match that of nBoxes')
end

if isscalar(onRatio) && nSignals > 1
    onRatio = repmat(onRatio, nSignals, 1);
else
    assert(numel(period) == nSignals, 'TAPAS:HUGE:Boxcar:InputSize', ...
        'Size of onRatio must match that of nBoxes')
end
assert(all(onRatio < 1) && all(onRatio > 0), 'TAPAS:HUGE:Boxcar:InputRange', ...
        'onRatio must be in range: 0 < onRatio < 1.')

if nargin < 5
    padding = zeros(nSignals, 2);
else
    if size(padding, 1) == 1
        padding = repmat(padding, nSignals, 1);
    end
    assert(size(padding, 1) == nSignals, 'TAPAS:HUGE:Boxcar:InputSize', ...
        'Number of rows in padding must match length of nBoxes')
end


%% generate signal
u = cell(1, nSignals);

for iSignal = 1:nSignals
    tMax = nBoxes(iSignal)*period(iSignal) + padding(iSignal, end);
    
    % amplitudes
    amp = repmat([1;0], 1, nBoxes(iSignal));
    
    % sample time points
    boxStarts = (0:nBoxes(iSignal)-1)*period(iSignal);
    boxDuration = [0; onRatio(iSignal)]*period(iSignal);
    grid = bsxfun(@plus, boxStarts, boxDuration);
    
    % query time points
    query = (0:dt:tMax)';
    
    % padding
    amp = [amp(:); 0];
    grid = [grid(:); tMax];
    if padding(iSignal, 1)
        query = [-fliplr(dt:dt:padding(iSignal, 1))'; query(:)];
        amp = [0; amp(:)];
        grid = [-padding(iSignal, 1); grid(:)];
    end
    % generate boxcar
    u{iSignal} = interp1(grid, amp, query, 'previous');

end

