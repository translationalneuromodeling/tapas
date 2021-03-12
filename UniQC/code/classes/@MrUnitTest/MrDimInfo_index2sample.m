function this = MrDimInfo_index2sample(this)
% Unit test for MrDimInfo index2sample and sample2index
%
%   Y = MrUnitTest()
%   run(Y, 'MrDimInfo_index2sample')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDimInfo_index2sample
%
%   See also MrUnitTest

% Author:   Saskia Bollmann
% Created:  2018-01-15
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


% construct MrDimInfo object from sampling points
dimInfo = this.make_dimInfo_reference(0);

% test array
arrayIndices = [58 14 142 35 2; 36 20 86 53 6; 10 25 6 23 18; ...
    171 290 361 38 430; 1 2 3 2 3]';
% convert index to samples
samples = dimInfo.index2sample(arrayIndices);
% convert samples to index
indeces = dimInfo.sample2index(samples);

% define expected solution
expSolution = arrayIndices;
% define actual solution
actSolution = indeces;

% verify whether expected and actual solution are identical
this.verifyEqual(actSolution, expSolution, 'absTol', 10e-7);
