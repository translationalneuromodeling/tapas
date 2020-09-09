function [ v, ff ] = tapas_ceode_update_euler( v, u, P, M, xtau, dt)
% [ v, ff ] = tapas_ceode_update_euler( v, u, P, M, xtau, dt)
%
% Computes an euler update step for the delayed euler integrator, i.e.
%
%    x(t + dt) = x(t) + dt * f(t-tau)
%
% INPUT 
%   v           mat          vectorized state value x(t) at time t
%   u           mat          value u(t) from driving input at time t
%   P           struct       parameter structure
%   M           struct       model specification
%   xtau        mat          delayed state matrix as returned from 
%                            tnudcm_compute_xtau.m
%   dt          mat          sampling rate of signal
% 
% OUTPUT
%   v           mat          vectorized state value x(t + dt) at time t + 1
%   ff          mat          f(t-tau) computed for delayed states
%
%
% -------------------------------------------------------------------------
%
% Author: Dario Schöbi
% Created: 2020-08-10
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS ceode Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------


f = str2func(M.f);
    
for j = 1 : numel(xtau)
    fx = full(f(v, u, P, M, xtau{j}));
    ff(j, :, :) = fx(j, :, :); 
end

ff = spm_vec(ff); 
v = v + dt * ff;

end
