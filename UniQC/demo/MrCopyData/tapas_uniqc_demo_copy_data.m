% Script demo_copy_data
% Shows functionality of MrCopyData for deep cloning and recursive operations
%
%  demo_copy_data
% 
% Since MrCopyData has no properties, we created a dummy class
% MrCopyDataTest with the following properties:
%   - numeric/string values, cells of nums/strings,
%   - MrCopyDataTest values and 
%   - cells of MrCopyDataTest
% to allow for a minimum comparison case of the recursive operations
% provided by MrCopyData. The MrCopyDataTest entries are recursive objects,
% but with one recursion depth less, i.e. at some point in the hierarchy,
% copydata_value and _cell will become empty
%
%   See also MrCopyData MrCopyDataTest

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-07-20
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

clear;
close all;
clc;
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create a deep MrCopyDataObject of given recursion depth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
nRecursionDepth = 2;
Y = MrCopyDataTest(nRecursionDepth);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform some unary operations for fun
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
Y.print
Y.find('MrCopyDataTest'); % recursive find, all props of this class
Y.get_nonempty_fields;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clone object and check comparison/diff works
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Z = Y.copyobj();
isequal(Y,Z) % should be the same...
dYZ = Y.diffobj(Z); % should have all-empty values...but resembles the recursive prop structure of the original object


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Manipulate clone and check comparison/diff works
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%
Z.copydata_value.clear(); % now sth is missing...
[dYZ, isYZEqual] = Y.diffobj(Z); %  "Y - Z", per property, chooses Y's prop where Z differs

Y.print_diff(Z);

isequal(Y,Z) % should be false...
isequal(Z,Y) % should be false...

% update all non-empty properties from other object
Z.update_properties_from(dYZ,1);
isequal(Y,Z) % TODO: bugfix! should be true again...
