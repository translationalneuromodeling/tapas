function imageGeom = make_imageGeometry_reference(this, varargin)
% create a MrImageGeometry reference object for unit testing
%
%   Y = MrUnitTest()
%   Y.make_imageGeometry_reference(do_save, fileName)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   make_imageGeometry_reference
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-17
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% check if created object should be saved for unit testing
if nargin > 1
    do_save = varargin{1};
else
    do_save = 0;
end

if nargin > 2
    fileName = varargin{2};
    makeFromFile = 1;
else
    makeFromFile = 0;
end

if makeFromFile
    imageGeom = MrImageGeometry(fileName);
    [~,name] = fileparts(fileName);
    % get classes path
    classesPath = tapas_uniqc_get_path('classes');
    % make full filename using date
    filename = fullfile(classesPath, '@MrUnitTest' , ...
        ['imageGeom-' name '.mat']);
else
    % get reference MrDimInfo object
    dimInfo = this.make_dimInfo_reference;
    % get reference MrAffineTransformation object
    affineTransformation = this.make_affineTransformation_reference;
    
    % set imageGeom properties
    imageGeom = MrImageGeometry(dimInfo, affineTransformation);
    
    % get classes path
    classesPath = tapas_uniqc_get_path('classes');
    % make full filename using date
    filename = fullfile(classesPath, '@MrUnitTest' , 'imageGeom.mat');
end
if do_save
    if exist(filename, 'file')
        prompt = 'Overwrite current MrImageGeometry constructor reference object? Y/N [N]:';
        answer = input(prompt, 's');
        if strcmpi(answer, 'N')
            do_save = 0;
        end
    end
end
if do_save
    save(filename, 'imageGeom');
end
