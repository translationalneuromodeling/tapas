function this = plot(this, module, varargin)
% Plots different aspects of MrSeries using MrImage-plot functions
%
%   Y = MrSeries()
%   Y.plot(inputs)
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%
% EXAMPLE
%   plot
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if ~exist('module', 'var')
    module = 'data';
end

switch module
    case {'data', 'mean', 'sd', 'snr', 'coeffVar', 'diffLastFirst'} 
        this.(module).plot(varargin{:});
    otherwise
        error('tapas:uniqc:MrSeries:InvalidPlotModule', ...
            'plotting module %s not implemented for MrSeries', ...
            module);
end
