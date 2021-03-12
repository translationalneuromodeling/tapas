function this = MrDataNd_select(this, testVariants)
% Unit test for MrDataNd select method.
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_select')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_select
%
%   See also MrUnitTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-09-03
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% construct reference MrDimInfo object
dimInfo = this.make_dimInfo_reference(0);
% select subset of sampling points to speed up unit test
dimInfo = dimInfo.select('t', 5:10);
% creat MrDataNd object
dataNd = MrDataNd(ones(dimInfo.nSamples), 'dimInfo', dimInfo);

% make selection struct for for two dimensions
selDim = [3, 4];

selDimChar = dimInfo.dimLabels(selDim);
selArray = {[10,13,16], 1};
for n = 1:numel(selDim)
    selection.(selDimChar{n}) = selArray{n};
end

switch testVariants
    case 'multipleDims'
        % nothing to do add here, just perform selection
        
        % select DataNd
        [actSolution, selectionIndexArray] = ...
            dataNd.select(selection);
        
        % make expected solution
        dataNd.data = dataNd.data(selectionIndexArray{:});
        dataNd.dimInfo = dataNd.dimInfo.select(selection);
        expSolution = dataNd;
        
    case 'invert'
        % set invert to ture
        selection.invert = true;
        % select DataNd
        [actSolution, selectionIndexArray] = ...
            dataNd.select(selection);
        
        % make expected solution
        dataNd.data = dataNd.data(selectionIndexArray{:});
        dataNd.dimInfo = dataNd.dimInfo.select(selection);
        expSolution = dataNd;
        
    case 'removeDims'
        % set remove dims to true
        selection.removeDims = true;
        
        % select DataNd
        [actSolution, selectionIndexArray] = ...
            dataNd.select(selection);
        
        % make expected solution
        dataNd.data = squeeze(dataNd.data(selectionIndexArray{:}));
        dataNd.dimInfo = dataNd.dimInfo.select(selection);
        expSolution = dataNd;
    case 'unusedVarargin'
        % make expected solution
        expSolution = {'giveBack', 'unusedVarargin'};

        % add unused varargin
        selection.(expSolution{1}) = expSolution{2};
        % select DataNd
        [~, ~, actSolution] = ...
            dataNd.select(selection);
        

        
end

% verify is actual solution matches expected solution
this.verifyEqual(actSolution, expSolution);

end