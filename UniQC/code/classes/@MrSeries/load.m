function this = load(this, fileName, varargin)
% Loads MrSeries from given folder or loads image files (.nii etc.) into MrSeries.data
%
%   Y = MrSeries()
%   Y.load(fileName)
%
% This is a method of class MrSeries.
%
% IN
%   fileName
%               '.mat',
%               'nii',
%               '.img'/'.hdr'
%               <data>  - 4D matrix
%               MrSeries is created from 4D-data file
%               <folderName> saved MrSeries is loaded from folder structure
% OUT
%
% EXAMPLE
%   load
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-03
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


isMatrix = ~isstr(fileName);

if isMatrix % assuming data matrix, loading via MrImage
    this.data.load(fileName, varargin);
else
    
    [fp, fn, ext] = fileparts(fileName);
    
    switch ext
        case {'.nii', '.img', '.hdr', '.mat', '.par', '.rec', '.cpx'}
            this.data.load(fileName, varargin);
        case '' % folder given where MrSeries was saved
            %sophisticated loading of whole MrSeries with its history of
            %processing steps
            if exist(fileName, 'dir')
                this.parameters.save.path = fileName;
                %somehow, all variable parameters are converted
                %into a cell, if varargin is given directly...
                this.restore('last', varargin{:}); % automatically restores last processing step
            else
                error('tapas:uniqc:MrSeries:InvalidPath', ...
                    'Path for MrSeries does not exist. Cannot restore.');
            end
    end
    this.name = regexprep(this.data.name, 'MrImage_', 'MrSeries_');
    this.data.name = 'data';
end