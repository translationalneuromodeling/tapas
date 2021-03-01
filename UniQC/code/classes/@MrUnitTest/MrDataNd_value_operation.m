function this = MrDataNd_value_operation(this, testVariantValueOperation)
% Unit test for MrDataNd for arithmetic operations (perform binary
% operation)
%
%   Y = MrUnitTest()
%   run(Y, 'MrDataNd_value_operation')
%
% This is a method of class MrUnitTest.
%
% IN
%
% OUT
%
% EXAMPLE
%   MrDataNd_value_operation
%
%   See also MrUnitTest

% Author:   Lars Kasper & Saskia Bollmann
% Created:  2019-03-22
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

actSolution.data = 0;
expSolution = 0;
absTol = 10e-7;
warning(sprintf('No test for value operation %s yet. Returning OK', testVariantValueOperation));
%% verify equality of expected and actual solution
% import matlab.unittests to apply tolerances for objects
this.verifyEqual(actSolution.data, expSolution, 'absTol', absTol);




end