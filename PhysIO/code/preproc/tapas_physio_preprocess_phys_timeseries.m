function ons_secs = tapas_physio_preprocess_phys_timeseries(ons_secs, sqpar, doNormalize)
% Preprocess raw cardiac/respiratory time series (e.g. amplitude
% normalization, padding for acquisition duration)
%
%   ons_secs = tapas_physio_preprocess_phys_timeseries(ons_secs)
%
% IN
%   ons_secs    raw time series (c(ardiac), r(espiratory), t(ime), cpulse
%               (from file))
%
% OUT
%   ons_secs    padded and normalized time series for preprocessing
%
% EXAMPLE
%   tapas_physio_preprocess_phys_timeseries
%
%   See also

% Author: Lars Kasper
% Created: 2016-02-28
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    doNormalize = 1;
end

hasCardiacData = ~isempty(ons_secs.c);
hasRespData = ~isempty(ons_secs.r);
hasDetectedCardiacPulses = ~isempty(ons_secs.cpulse);
hasAcquisitionCodes = ~isempty(ons_secs.acq_codes);

%% Normalize cardiac/respiratory time series to max 1

if doNormalize
    if hasCardiacData
        maxAbsC = max(abs(ons_secs.c));
        ons_secs.c_scaling = maxAbsC ;
        ons_secs.c = ons_secs.c/maxAbsC;
    end
    
    if hasRespData
        maxAbsR = max(abs(ons_secs.r));
        ons_secs.r_scaling = maxAbsR ;
        ons_secs.r = ons_secs.r/maxAbsR;
    end
    
end

% since resampling might have occured during read-in, dt is recalculated
ons_secs.dt = ons_secs.t(2) - ons_secs.t(1);


%% Padding of cardiac/respiratory time series, if start too late or duration too short

durationScan = sqpar.TR*(sqpar.Nscans+sqpar.Ndummies);

t = ons_secs.t;

tStartScan = 0;
if t(1) > tStartScan
    nSamples = ceil(-t(1)/ons_secs.dt);
    paddingStart = zeros(nSamples, 1);
    
    ons_secs.t = [(nSamples:-1:1)'*ons_secs.dt + t(1);t];
    
    if hasCardiacData
        ons_secs.c = [paddingStart; ons_secs.c];
    end
    
    if hasRespData
        ons_secs.r = [paddingStart; ons_secs.r];
    end
    
    if hasAcquisitionCodes
        ons_secs.acq_codes = [paddingStart; ons_secs.acq_codes];
    end
end

t = ons_secs.t;
durationPhysLog = t(end) - t(1);

if durationPhysLog < durationScan
    
    nSamples = ceil((durationScan - durationPhysLog)/ons_secs.dt);
    paddingEnd = zeros(nSamples, 1);
    
    ons_secs.t = [t; (1:nSamples)'*ons_secs.dt + t(end)];
    if hasCardiacData
        ons_secs.c = [ons_secs.c; paddingEnd];
    end
    
    if hasRespData
        ons_secs.r = [ons_secs.r; paddingEnd];
    end
    
    if hasAcquisitionCodes
        ons_secs.acq_codes = [ons_secs.acq_codes; paddingEnd];
    end
end


%% Remove onset time of log file to simplify plotting
tStartLog = t(1);
ons_secs.t_start = tStartLog;

ons_secs.t = ons_secs.t - tStartLog;

if hasDetectedCardiacPulses
    ons_secs.cpulse = ons_secs.cpulse - tStartLog;
end
