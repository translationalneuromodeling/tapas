function this = save(this, filename)
% saves the data in MrRoi
%
%   Y = MrRoi()
%   Y.save(inputs)
%
% This is a method of class MrRoi.
%
% IN
%
% OUT
%
% EXAMPLE
%   save
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



if nargin <2
   filename = fullfile(this.parameters.save.path, this.parameters.save.fileName);
end

% no data, no saving...
if isempty(this.data)
    fprintf('No data in MrImage %s; file %s not saved\n', this.name, ...
        filename);
    
else
    [fp] = fileparts(filename);
    
    if ~isempty(fp) && ~exist(fp, 'dir')
        mkdir(fp);
    end
    
    data = this.data;
    save(filename, 'data');
end

end