function [pks,locs] = tapas_physio_findpeaks(X,varargin)
% Finds local peaks in the data. Wrapper for Matlab's findpeaks, 
% with compatibility alternative, if no signal processing toolbox available
%
%  [pks,locs] = tapas_physio_findpeaks(X,varargin)
%
% IN
%   X       input data vector
%   varargin property name/value pairs, the following options are available
%
%   'minPeakHeight'
%   'minPeakDistance'
%   'threshold'     local elevation compared to neighbours
%   'nPeaks'
%   'sorstr'        'ascend' or 'descend'; default: no sorting, order of
%                   occurence
%   
% OUT
%   pks     height of local peaks (maxima)
%   locs    location indices of peaks
%
% EXAMPLE
%   tapas_physio_findpeaks
%
%   See also
 
% Author:   Lars Kasper
% Created:  2019-02-01
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
if exist('findpeaks', 'file')
    [pks,locs] = findpeaks(X,varargin{:}); 
else
    % previously:
    % [pks,locs] = tapas_physio_findpeaks_compatible(X,varargin{:});
    tapas_physio_log(sprintf(...
        ['tapas_physio_findpeaks_compatible is no longer\n' ...
        'distributed with this toolbox. Instead, we rely on the "findpeaks"\n' ... 
        'function included in Matlab''s Signal Processing Toolbox.']), [], 2)
end