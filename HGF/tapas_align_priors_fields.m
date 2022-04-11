function c = tapas_align_priors_fields(c)
% Aligns parameter fields with the explicit prior definitions with the 
% content of the vectors c.priormus and c.priorsas (vice-versa of function 
% 'tapas_align_priors.m').
%
%   Example:
%   >> c_prc = tapas_ehgf_binary_config;
%   >> c_prc.priormus(13) = -2;
%   >> c_prc = tapas_align_priors_fields(c_prc)
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Get fieldnames. If a name ends on 'mu', that field defines a prior mean.
% If it ends on 'sa', it defines a prior variance.
names = fieldnames(c);
pm = 0;
ps = 0;

% Loop over fields and overwrite fiels whose name ends on 'mu or 'sa'
for i = 1:length(names)
    if regexp(names{i}, 'mu$')
        c.(names{i}) = c.priormus(pm+1:pm+length(c.(names{i})));
        pm = pm+length(c.(names{i}));
    elseif regexp(names{i}, 'sa$')
        c.(names{i}) = c.priorsas(ps+1:ps+length(c.(names{i})));
        ps = ps+length(c.(names{i}));
    end
end

end
