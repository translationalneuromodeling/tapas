function stats = regionprops3(this, varargin)
% Estimates properties from a binary image via connected components.
%
%   Y = MrImage()
%   Y.regionprops3('conn', connectivity, 'props', {'Prop1', 'Prop2'})
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   stats = Y.regionprops3('conn', 6, 'properties', {'all'})
%
%   See also MrImage bwconncomp regionprops3

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2020-05-22
% Copyright (C) 2020 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% parse input arguments
defaults.conn = 6;
defaults.props = {'basic'};
[args, ~] = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% compute connected components
CC = bwconncomp(this.data, conn);
% extract properties
stats = regionprops3(CC, props{:});

end