classdef MrPlots < MrCopyData
%Abstract class to create fancy plots from multiple other class objects
% (e.g. boxplots of rois)
%
%
% EXAMPLE
%   MrPlots
%
%   See also

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-01
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


properties
 % COMMENT_BEFORE_PROPERTY
end % properties
 
 
methods

% Constructor of class
function this = MrPlots()
end

% NOTE: Most of the methods are saved in separate function.m-files in this folder;
%       except: constructor, delete, set/get methods for properties.

end % methods


methods(Static)
    
function boxplot
    disp('has to be implemented by Saskia...');
end

end

end
