function phase_expansion = tapas_physio_get_fourier_expansion(phase, order)
% does a cosine/sine expansion of a given phase up to the specified order.
%
% USAGE:
%   phase_expansion = tapas_physio_get_fourier_expansion(phase, order)
%
% INPUT
%   phase
%   order
% 
% OUTPUT
%   phase_expansion
%
% One expansion term is one column of the output with cos/sin sorted
% by order, alternating
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%
if (order < 1) %well, not correct in a strict sense, but more convenient than returning a constant
    phase_expansion = []
else
    phase_expansion=zeros(size(phase,1),order);
    for i=1:order
        phase_expansion(:,2*i-1)=cos(i*phase);
        phase_expansion(:,2*i)=sin(i*phase);
    end
    phase_expansion(find(isnan(phase_expansion)))=0;
end
