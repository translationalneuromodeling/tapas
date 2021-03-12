function this = MrDimInfo_load_from_mat(this, testCaseLoadMat)
%ONE_LINE_DESCRIPTION
%
%   Y = MrUnitTest()
%   Y.MrDimInfo_load_from_mat(inputs)
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_load_from_mat
%
%   See also MrUnitTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-15
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
dimInfo = this.make_dimInfo_reference(0);
% save parameters
dataPath = tapas_uniqc_get_path('data');
folderName = fullfile(dataPath, 'temp_MrDimInfo_load_from_mat');
% TODO: replace with proper exist check
[~, ~] = mkdir(folderName);
fileName = fullfile(folderName, [testCaseLoadMat, '.mat']);
switch testCaseLoadMat
    
    case 'checkTempDir'
        % expect solution is an empty directory ( which contains '.' and
        % '..' only)
        expSolution(1).name = '.';
        expSolution(2).name = '..';
        
        % verify first that temp dir is empty
        dirContent = dir(fullfile(folderName, '*'));
        [actSolution(1:numel(dirContent)).name] = dirContent.name;
        
    case 'oneVar'
        
        % expected solution
        expSolution = dimInfo;
        
        % actual solution
        warning('off', 'MATLAB:structOnObject');
        anyVarName = struct(dimInfo); %#ok<NASGU>
        warning('on', 'MATLAB:structOnObject');
        
        % save dimInfo as struct
        save(fileName, 'anyVarName');
        
        % now load dimInfo
        actSolution = MrDimInfo(fileName);
        
        % delete files
        delete(fileName);
        rmdir(folderName);
        
    case 'objectAsStruct'
        
        % expected solution
        expSolution = dimInfo;
        
        % actual solution
        warning('off', 'MATLAB:structOnObject');
        objectAsStruct = struct(dimInfo); %#ok<NASGU>
        warning('on', 'MATLAB:structOnObject');
        
        % save dimInfo as struct
        save(fileName, 'objectAsStruct');
        
        % now load dimInfo
        actSolution = MrDimInfo(fileName);
        
        % delete files
        delete(fileName);
        rmdir(folderName);
        
    case 'className'
        
        % expected solution
        expSolution = dimInfo;
        
        % actual solution
        warning('off', 'MATLAB:structOnObject');
        dimInfo = struct(dimInfo); %#ok<NASGU>
        warning('on', 'MATLAB:structOnObject');
        
        % save dimInfo as struct
        save(fileName, 'dimInfo');
        
        % now load dimInfo
        actSolution = MrDimInfo(fileName);
        
        % delete files
        delete(fileName);
        rmdir(folderName);
        
    case 'noMatch'
        
        % expected solution - empty MrDimInfo
        % should verrify whether arning is issued, but couldn't figure out
        % how to get a function handle from a dynamic object method :?
        expSolution = MrDimInfo;
        
        % two dummy variables
        var1 = 1;
        var2 = 2;
        
        % save dimInfo as struct
        save(fileName, 'var1', 'var2');
        
        % now load dimInfo
        actSolution = MrDimInfo(fileName);
        
        % delete files
        delete(fileName);
        rmdir(folderName);
        
    case 'tooManyMatch'
        
        % expected solution - empty MrDimInfo
        % should verrify whether arning is issued, but couldn't figure out
        % how to get a function handle from a dynamic object method :?
        expSolution = MrDimInfo;
        
        % two dummy variables
        objectAsStruct = 1;
        dimInfo = 2;
        
        % save dimInfo as struct
        save(fileName, 'objectAsStruct', 'dimInfo');
        
        % now load dimInfo
        actSolution = MrDimInfo(fileName);
        
        % delete files
        delete(fileName);
        rmdir(folderName);
        
    case 'withVarName'
        
        % expected solution
        expSolution = dimInfo;
        
        % actual solution
        warning('off', 'MATLAB:structOnObject');
        var1 = struct(dimInfo); %#ok<NASGU>
        warning('on', 'MATLAB:structOnObject');
        var2 = 2;
        
        % save dimInfo as struct
        save(fileName, 'var1', 'var2');
        
        % now load dimInfo
        % this does not work via the constructor, but only using load
        % explicitely
        
        actSolution = MrDimInfo();
        actSolution.load(fileName, 'var1');
        
        % delete files
        delete(fileName);
        rmdir(folderName);
end

% verify whether expected and actual solution are identical
% Note: convert to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');


end