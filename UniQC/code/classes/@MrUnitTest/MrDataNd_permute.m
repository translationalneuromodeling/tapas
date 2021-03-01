function this = MrDataNd_permute(this)
% Unit test for MrDataNd.permute()
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_permute')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_permute
%
%   See also MrUnitTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-28
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% construct MrDimInfo object from sampling points
dimInfo = this.make_dimInfo_reference(0);
% select subset of sampling points to speed up unit test
dimInfo = dimInfo.select('t', 5:10);
% create MrDataNd object
dataNd = MrDataNd(rand(dimInfo.nSamples), 'dimInfo', dimInfo);

% define permutation
permutation = [3 1 4 2 5];

% create expected solution
expSolution = dataNd.copyobj;
% permute data
expSolution.data = permute(expSolution.data, permutation);
% permute dimInfo
expSolution.dimInfo.permute(permutation);
% update info field
expSolution.info{end+1,1} = sprintf('permute(this, [%s]);', sprintf('%d ', ...
    permutation));

% permute / define actual solution
actSolution = dataNd.permute([3 1 4 2 5]);

% verify whether expected and actual solution are identical
% Note: convert to struct, since the PublicPropertyComparator (to allow
% nans to be treated as equal) does not compare properties of objects that
% overload subsref

warning('off', 'MATLAB:structOnObject');
this.verifyEqual(struct(actSolution), struct(expSolution), 'absTol', 10e-7);
warning('on', 'MATLAB:structOnObject');



end
