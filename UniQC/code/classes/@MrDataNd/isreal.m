function isDataReal = isreal(this)
% Checks whether all data elements in image are real
%
%   Y = MrImage();
%   isDataReal = isreal(Y)
%   
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%   isDataReal            true, if all elements in MrImage.data are real
%                         false, otherwise (i.e. complex entries)
%
% EXAMPLE
%   isDataReal = isreal(Y)
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-12-01
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


isDataReal = isreal(this.data);