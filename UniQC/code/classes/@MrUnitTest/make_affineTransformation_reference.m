function affineTrafo = make_affineTransformation_reference(~, varargin)
% create a affineTransformation reference object for unit testing
%
%   Y = MrUnitTest()
%   Y.make_affineTransformation_reference(do_save, fileName)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   make_affineTransformation_reference
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2017-11-30
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

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
    affineTrafo = MrAffineTransformation(fileName);
    [~,name] = fileparts(fileName);
    % get classes path
    classesPath = tapas_uniqc_get_path('classes');
    % make full filename using date
    filename = fullfile(classesPath, '@MrUnitTest' , ...
        ['affineTrafo-' name '.mat']);
else
    affineTrafo = MrAffineTransformation(...
        'offcenter_mm', [25, 30, 11], 'rotation_deg', [3 -6 10], ...
        'shear', [0.2 3 1], 'scaling', [1.3 1.3 1.25]);
    
    % get classes path
    classesPath = tapas_uniqc_get_path('classes');
    % make full filename using date
    filename = fullfile(classesPath, '@MrUnitTest' , ['affineTrafo.mat']);
end

if do_save
    if exist(filename, 'file')
        prompt = 'Overwrite current MrAffineTransformation constructor reference object? Y/N [N]:';
        answer = input(prompt, 's');
        if strcmpi(answer, 'N')
            do_save = 0;
        end
    end
end
if do_save
    save(filename, 'affineTrafo');
end