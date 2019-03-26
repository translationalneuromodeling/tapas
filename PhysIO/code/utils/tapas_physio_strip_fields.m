function tapas_physio_strip_fields(opts)
% Strips fields of a structure into workspace variables of calling function
%
%   tapas_physio_strip_fields(opts)
%
% IN
%   opts    structure variable
%
% OUT
%
% SIDE EFFECTS
%   creates variables with field names in workspace of calling function
%
% EXAMPLE
%   opts.myString = 'Hello';
%   opts.coolArray = rand(100);
%   strip_fields(opts);
%
%   => command "whos" will show the following workspace variables 
%      (stripped fields of structure opts):
%
%       opts        1x1     struct
%       myString    1x5     char
%       coolArray   100x100 double
%
%   See also propval

% Author: Lars Kasper
% Created: 2013-11-13
%
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



optsArray = fields(opts);
nOpts = length(optsArray);
for iOpt = 1:nOpts
    assignin('caller', optsArray{iOpt}, opts.(optsArray{iOpt}));
end