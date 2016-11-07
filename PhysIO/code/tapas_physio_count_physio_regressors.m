function nPhysioRegressors = tapas_physio_count_physio_regressors(physio)
% Returns number of physiological regressors created, given model
% specification
%
% IN
%   physio  physio-structure, See also tapas_physio_new
%
% OUT
% nPhysioRegressors     number of physiological regressors, e.g.
%
% EXAMPLE
%   tapas_physio_report_contrasts
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-10-16
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 464 2014-04-27 11:58:09Z kasperla $

model = physio.model;

hasRetroicor = ~isempty(regexpi(model.type, 'RETROICOR'));
hasHrv = ~isempty(regexpi(model.type, 'HRV'));
hasRvt = ~isempty(regexpi(model.type, 'RVT'));

nPhysioRegressors = 0;

if hasHrv
    nPhysioRegressors = nPhysioRegressors + 1;
end

if hasRvt
    nPhysioRegressors = nPhysioRegressors + 1;
end

if hasRetroicor
    order = model.order;
    nPhysioRegressors = nPhysioRegressors + ...
        2*order.c + ...
        2*order.r + ...
        4* order.cr;
end

end
