function tests = tapas_physio_conv_test()
% Tests whether current PhysIO's own convolution fulfills specifications,
% i.e. 
%   1) is causal (i.e., response starts only at t=0 of the input) in the
%      'causal' setting, and 
%   2) is symmetric in the symmetric setting (i.e.,
%      response function centered around Dirac input) NOTE: This is what
%      Matlab's conv function does in the 'same' setting
%   3) as (2) but this time indeed using Matlab's conv(..., 'same') for
%      reference solution
%
%    tests = tapas_physio_conv_test()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_conv_test
%
%   See also

% Author:   Sam Harrison
% Created:  2019-07-17
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

tests = functiontests(localfunctions);
end

function test_conv_causal(testCase)
%% Tests causal convolution via an impulse response

impulse  = [0 0 0 0 1 0 0 0 0];
filter   = [1 2 3];
solution = [0 0 0 0 1 2 3 0 0];

verifyEqual(testCase, ...
    tapas_physio_conv(impulse, filter, 'causal', 'zero'), ...
    solution);

end

function test_conv_symmetric(testCase)
%% Tests non-causal convolution (symmetric response around input) 
% via an impulse response

impulse  = [0 0 0 0 1 0 0 0 0];
filter   = [1 2 3];
solution = [0 0 0 1 2 3 0 0 0];

verifyEqual(testCase, ...
    tapas_physio_conv(impulse, filter, 'symmetric', 'zero'), ...
    solution);

end

function test_conv_symmetric_matlab_conv_same(testCase)
%% Tests non-causal convolution vs Matlab's conv(...,'same')

impulse  = [0 0 0 0 1 0 0 0 0];
filter   = [1 2 3];
solution =  conv(impulse, filter, 'same');  % [0 0 0 1 2 3 0 0 0];

verifyEqual(testCase, ...
    tapas_physio_conv(impulse, filter, 'symmetric', 'zero'), ...
   solution);

end