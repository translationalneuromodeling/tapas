function [y] = tapas_ceode_int_euler(P, M, U)
% [y] = tapas_ceode_int_euler(P, M, U)
% 
% Integration of a (delayed) dynamical system (convolution based DCM)
% based on the continuous extension for ODE methods. ODE step follows a 
% forward Euler scheme:
%
%   x_{n+1} = x_{n} + dt .* f(xhat_{n-taun})
%
% Also see ceode_compute_xtau.m. 
% Adapted from spm_fx_erp.m (original function lincense below).
%
%
% INPUT
%   P           struct          parameter structure
%   M           struct          model specification
%   U           struct          design specification
%
% OUTPUT
%   y           mat             Integrated activity in sensor space
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
%
% ORIGINAL FUNCTION LICENSE
%
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
% Karl Friston
% $Id: spm_int_L.m 6110 2014-07-21 09:36:13Z karl $
%--------------------------------------------------------------------------
 
if isfield(M, 'intstep')
    N = ceil(U.dt / M.intstep);
else
    N = ceil(U.dt / 1E-3);
end

% If subsampling is needed, re-evaluate driving input
 if N > 1   
    U.dt = U.dt / N;
    U.u = feval(M.fu, U.dt .* [1 : M.ns*N], P, M);
 end
 
% convert U to U.u if necessary
%--------------------------------------------------------------------------
if ~isstruct(U), u.u = U; U = u;   end
try, dt = U.dt;  catch, dt = 1;    end

% Initial states and inputs
%--------------------------------------------------------------------------
try
    x   = M.x;
catch
    x   = sparse(0,1);
    M.x = x;
end

try
    u   = U.u(1,:);
catch
    u   = sparse(1,M.m);
end

% add [0] states if not specified
%--------------------------------------------------------------------------
try
    f   = spm_funcheck(M.f);
catch
    f   = @(x,u,P,M) sparse(0,1);
    M.n = 0;
    M.x = sparse(0,0);
    M.f = f;
end

 
% output nonlinearity, if specified
%--------------------------------------------------------------------------
try
    g   = spm_funcheck(M.g);
catch
    g   = @(x,u,P,M) x;
    M.g = g;
end


% dx(t)/dt and Jacobian df/dx and check for delay operator
%--------------------------------------------------------------------------
if nargout(f) >= 3
    [fx, ~, ~] = f(x,u,P,M);
    
elseif nargout(f) == 2
    [fx, dfdx]   = f(x,u,P,M);
    
else
    dfdx  = spm_cat(spm_diff(f,x,u,P,M,1)); 
end

 
% initialize states, history and delays
%==========================================================================
xhist = zeros(length(spm_vec(x)), M.ns*N);
v = spm_vec(x);
xhist(:, 1) = v;
itau = tapas_ceode_compute_itau(P, M, U);

% integrate
%==========================================================================
for i = 1 : size(U.u,1)
    
    % input
    %----------------------------------------------------------------------
    u  = U.u(i,:);
    
    xtau = tapas_ceode_compute_xtau(i, xhist, itau);       
    xhist(:, i + 1) = ...
        tapas_ceode_update_euler(xhist(:, i), u, P, M, xtau, dt);
  
 
end

% output - implement g(x)
%------------------------------------------------------------------
% back to the original trace length after subsampling

idx = 1:N:size(xhist,2)-1; 
y = zeros(size(xhist,1),numel(idx));
for j = 1:length(idx)
    y(:, j) = g(xhist(:,idx(j)), u, P, M);
end

% transpose
%--------------------------------------------------------------------------
y      = real(y');

