function LOCS = tapas_physio_create_LOCS_from_VOLLOCS(VOLLOCS, t, sqpar)
% Compute slice scan events as locations in time vector (LOCS) from volume
% scan locations and sequence parameters (TR, nSlices)
%
% LOCS = tapas_physio_create_LOCS_from_VOLLOCS(VOLLOCS, t, sqpar);
%
%
% IN
%   NOTE: The detailed description of all input structures can be found as
%   comments in tapas_physio_new
%
%   VOLLOCS - index locations in time vector (of physiological recordings),
%                             when volume scan events started
%   t       - time vector of phys time course
%
%   sqpar   - sequence timing parameters, used for computation
%             of scan events from 'nominal' timing
%           .Nslices        - number of slices per volume in fMRI scan
%           .TR             - repetition time in seconds
%           .time_slice_to_slice - time between the acquisition of 2 subsequent
%                             slices; typically TR/Nslices or
%                             minTR/Nslices, if minimal temporal slice
%                             spacing was chosen
%
% OUT
%   LOCS    - locations in time vector, when slice or volume scan
%             events started
%
%   See also tapas_physio_new tapas_physio_main_create_regresssors

% Author: Lars Kasper
% Created: 2013-08-23
% Copyright (C) 2016 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


NallVols        = sqpar.Ndummies+sqpar.Nscans;
Nslices         = sqpar.Nslices;
TR              = sqpar.TR;
LOCS            = NaN(NallVols*Nslices,1);

%default for equidistantly spaced slices
if isempty(sqpar.time_slice_to_slice)
    sqpar.time_slice_to_slice = TR/Nslices;
end

iFirstValidVolume = find(~isnan(VOLLOCS), 1);
tRef = t(VOLLOCS(iFirstValidVolume));


for n = iFirstValidVolume:NallVols
    
    % first, restrict search interval to within TR
    LOCVOLSTART = VOLLOCS(n);
    
    if n == NallVols
        LOCVOLEND = numel(t);
    else
        LOCVOLEND = VOLLOCS(n+1);
    end
    
    % Then, find slice-by-slice nearest neighbour of actual and
    % reference time
    tSearchInterval = t(LOCVOLSTART:LOCVOLEND) - ...
        (tRef + sqpar.TR*(n-iFirstValidVolume));
    
    % Note: Why not the same as other slices? Some jitter problem...
    % if TR is not given with high enough precision, therefore reset for
    % each volume!
    LOCS((n-1)*Nslices + 1) = LOCVOLSTART; 
    for s = 2:sqpar.Nslices
        [tmp, RELATIVELOC] = min(abs(tSearchInterval - ...
            sqpar.time_slice_to_slice*(s-1)) );
        
        LOCS((n-1)*Nslices + s) = LOCVOLSTART + RELATIVELOC - 1;
    end
end