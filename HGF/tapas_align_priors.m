function c = tapas_align_priors(c)
% Aligns c.priormus and c.priorsas with the contents of the explicit prior
% definitions.
%
%   Example:
%   >> c_prc = tapas_ehgf_config;
%   >> c_prc.ommu(2) = -2;
%   >> c_prc = tapas_align_priors(c_prc)
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Initialize new prior mean vector
priormus = [];
% Initialize new prior variance vector
priorsas = [];

% Get fieldnames. If a name ends on 'mu', that field defines a prior mean.
% If it ends on 'sa', it defines a prior variance.
names = fieldnames(c);

% Loop over fields
for i = 1:length(names)
    if regexp(names{i}, 'mu$')
        priormus = [priormus, c.(names{i})];
    elseif regexp(names{i}, 'sa$')
        priorsas = [priorsas, c.(names{i})];
    end
end

% Replace old vectors with newly aligned ones
c.priormus = priormus;
c.priorsas = priorsas;

end
