function this = load_data(this)
%ONE_LINE_DESCRIPTION
%
%   Y = MrRoi()
%   Y.load_data(inputs)
%
% This is a method of class MrRoi.
%
% IN
%
% OUT
%
% EXAMPLE
%   load_data
%
%   See also MrRoi

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-21
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% get filename
filename = fullfile(this.parameters.save.path, this.parameters.save.fileName);
% load data
data = struct2cell(load(filename, 'data'));
this.data = data{:};