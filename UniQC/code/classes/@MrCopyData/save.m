function this = save(this, fileName)
% saves object as struct (to be readable if code is not in path or
% outdated) in .mat-file , variable named 'objectAsStruct'
%
%   Y = MrCopyData()
%   Y.save(fileName)
%
% This is a method of class MrCopyData.
%
% IN
%
% OUT
%
% EXAMPLE
%   save
%
%   See also MrCopyData  MrCopyData.load

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-14
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% save dimInfo for later recovery of absolute indices (e.g.
% which coil or echo time)
warning('off', 'MATLAB:structOnObject');
objectAsStruct = struct(this);
warning('on', 'MATLAB:structOnObject');

[~,~] = mkdir(fileparts(fileName));
save(fileName, 'objectAsStruct');