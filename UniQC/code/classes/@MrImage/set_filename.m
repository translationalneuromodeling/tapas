function set_filename(this, filename)
% Takes input filename and sets parameters.save.path/fileName accordingly
%
%   Y = MrImage()
%   Y.set_filename(filename)
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y.set_filename('newFolder/newFile.nii');
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-12-10
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


[fp, fn, ext]                   = fileparts(filename);
this.parameters.save.path       = fp;
this.parameters.save.fileName   = [fn ext];