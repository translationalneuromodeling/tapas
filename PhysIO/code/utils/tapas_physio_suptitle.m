function tapas_physio_suptitle(titleString)
% if Biodemo toolbox not installed, use simple title instead of suptitle
%
%   tapas_physio_suptitle(titleString)
%
% IN
%   titleString     that should be displayed on top of figure with multiple
%                   subplots
% OUT
%
% EXAMPLE
%   tapas_physio_suptitle
%
%   See also
 
% Author:   Lars Kasper
% Created:  2019-01-31
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
if exist('suptitle', 'file')
    suptitle(titleString);
else
    title(titleString);
end
