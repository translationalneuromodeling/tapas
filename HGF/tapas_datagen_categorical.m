function u = tapas_datagen_categorical
% This function generates categorical input data for the hgf_categorical model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2015 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% First set of outcomes
u = mnrnd(1, [0.8, 0.1, 0.1], 64);

% Second set of outcomes
u = [u; mnrnd(1, [1/3, 1/3, 1/3], 64)];

% Third set of outcomes
u = [u; mnrnd(1, [0.1, 0.1, 0.8], 64)];

% Add next set of outcomes (...or don't)

% Turn u into a single column of natural numbers indicating outcome category
u = sum(u*diag([1 2 3]),2);
