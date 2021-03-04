function [this, saveFileName] = save(this, varargin)
% Saves 4D MrImage in single file.
%
%   Y = MrImageSpm4D()
%   Y.save('fileName', fileName)
%
% This is a method of class MrImageSpm4D.
%
% IN
%   fileName    string: default via get_filename
% OUT
%
% EXAMPLE
%   save
%
%   See also MrImageSpm4D MrDataNd.save MrDataNd.get_filename

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-23
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


defaults.fileName = this.get_filename();
args = tapas_uniqc_propval(varargin, defaults);

tapas_uniqc_strip_fields(args);

saveFileName = this.write_single_file(fileName);
end