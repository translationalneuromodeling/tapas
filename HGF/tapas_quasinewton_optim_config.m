function c = tapas_quasinewton_optim_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the Broyden, Fletcher, Goldfarb and Shanno (BFGS)
% quasi-Newton optimization algorithm
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Config structure
c = struct;

% Algorithm name
c.algorithm = 'BFGS quasi-Newton';

% Verbosity
c.verbose   = false;

% Options for optimization
c.tolGrad = 1e-3;
c.tolArg  = 1e-3;
c.maxStep = 1;
c.maxIter = 100;
c.maxRegu = 16;
c.maxRst  = 10;

% Algorithm filehandle
c.opt_algo = @tapas_quasinewton_optim;

return;
