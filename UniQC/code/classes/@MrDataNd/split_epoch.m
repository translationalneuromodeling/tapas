function epochedY = split_epoch(this, onsetTrials, newPeriStimulusOnsets)
%Splits time series in epochs using onset trial times and desired bin times
%
%   Y = MrDataNd()
%   epochedY = Y.split_epoch(onsetTrials, newPeriStimulusOnsets)
%
% This is a method of class MrDataNd.
% It first shifts the time series of all voxels according s.t. the trial
% onset of each trial coincided with one volume (using slice timing correction
% i.e., Fourier Interpolation), and then selects the following volumes as
% part of the trial.
%
% TODO: interpolation to higher grid sampling rate of desired PST bins
%
% IN
%   onsetTrials [1, nTrials] onset times (same unit as in dimInfo.t) for
%                            each trial
%   newPeriStimulusOnsets
%               [1, nTimeBins] onset times of desired peristimulus trial bins
%                          relative to trial onset = 0
%                   OR
%               [1,1], number of volumes after stimulus onset used (spaced
%                      by TR
% OUT
%   epochedY    [nDim1, ..., nDimN-1, nTimeBins, nTrials]
%                       new MrDataNd instance with one dimension more than
%                       input Y. time dimension is reduced (only number of
%                       scans after stimulus onset that are required by bin
%                       times), but an extra dimension holding the trial
%                       index is created.
%
% EXAMPLE
%   split_epoch
%
%   See also MrDataNd MrDataNd.shift_timeseries
 
% Author:   Andreea Diaconescu & Lars Kasper
% Created:  2019-03-20
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 2
    newPeriStimulusOnsets = 10;
end

TR  = this.dimInfo.t.samplingWidths;
% create time vector for new bins
if numel(newPeriStimulusOnsets) == 1
    newPeriStimulusOnsets = (0:(newPeriStimulusOnsets-1))*TR;
end

nVolumesPerTrial = numel(newPeriStimulusOnsets);

%% First, create shifts of the time series by slice timing correction
% , s.t. stimulus onsets coincide with one volume onset
onsetScans = this.dimInfo.t.samplingPoints{1};
nTrials = numel(onsetTrials);

% error, if requested re-sampled bins are outside data interval (no
% extrapolation allowed!)
if min(onsetTrials) < min(onsetScans)
    error('tapas:uniqc:MrDataNd:OnsetBinBeforeFirstScan', ...
        'min trial onset (%f) before first scan onset(%f %s) in MrDataNd (%s); no extrapolation allowed!', ...
        min(onsetTrials), min(onsetScans), this.dimInfo.t.units{1}, this.name);
elseif max(onsetTrials) + max(newPeriStimulusOnsets) > max(onsetScans)
    error('tapas:uniqc:MrDataNd:LastBinAfterLastScan', ...
        ['requested bins outside data range: resampled last bin onset (%f) after last ' ...
        'scan onset (%f %s) requested in MrDataNd (%s); no extrapolation allowed!'], ...
        max(onsetTrials) + max(newPeriStimulusOnsets), max(onsetScans), ...
        this.dimInfo.t.units{1}, this.name);
end

shiftedY = cell(nTrials,1);
fprintf('\nEpoching trial %04d/%04d', 0, nTrials);
for iTrial = 1:nTrials
    fprintf('\b\b\b\b\b\b\b\b\b%04d/%04d', iTrial, nTrials);
    onsetTrial = onsetTrials(iTrial);
    idxFirstVolumeAfterTrialOnset = find(onsetScans - onsetTrial >=0, 1, 'first');
    dt = onsetScans(idxFirstVolumeAfterTrialOnset) - onsetTrial;
    shiftedY{iTrial} = this.shift_timeseries(dt); % shift backwards so that t=0 becomes a volume
    shiftedY{iTrial} = shiftedY{iTrial}.select('t', idxFirstVolumeAfterTrialOnset + [0:(nVolumesPerTrial-1)]);
end
fprintf('\n');

% create relative time vector instead of absolute one
for iTrial = 1:nTrials
   shiftedY{iTrial}.dimInfo.set_dims('t', 'samplingPoints', newPeriStimulusOnsets); 
end

epochedY = shiftedY{1}.combine(shiftedY, 'trials');