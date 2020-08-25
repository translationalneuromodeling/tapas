function [f,J,D] = tapas_ceode_fx_cmc(x, u, P, M, xtau)
% [f,J,D] = tapas_ceode_fx_cmc(x, u, P, M, xtau)
% 
% Computation of the dynamical (state) equation for a neural mass model
% (canonical microcircuit). Compatibile with continuous extension for ODE
% methods. Adapted from spm_fx_cmc.m (original function license below).
% Function output signature kept the same to ensure compatibility.
%
%
% INPUT
%   x           mat             vector of current state activity
%   u           mat             vector of current driving input
%   P           struct          parameter structure
%   M           struct          model specification
%   xtau        mat             vector of delayed state activity
%
% OUTPUT
%   f           mat             matrix of dx/dt
%   J           mat             Jacobian of the system df/dx. Fixed to 0
%                               since irrelevant for this integration
%                               scheme.
%   D           mat             matrix of delays. Empty since irrelevant
%                               for this integration scheme.

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
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging
% 
% Karl Friston
% $Id: spm_fx_cmc.m 6720 2016-02-15 21:06:55Z karl $
%--------------------------------------------------------------------------


% get dimensions and configure state variables
%--------------------------------------------------------------------------
if nargin < 5
    xtau = x; 
end 

% get dimensions and configure state variables
%--------------------------------------------------------------------------
x  = spm_unvec(x,M.x);            % neuronal states
xtau = spm_unvec(xtau, M.x);
n  = size(x,1);                   % number of sources

% [default] fixed parameters
%--------------------------------------------------------------------------
E  = [1 1/2 1 1/2]*200;           % extrinsic (forward and backward)
G  = [4 4 8 4 4 2 4 4 2 1]*200;   % intrinsic connections
T  = [2 2 16 28];                 % synaptic time constants
R  = 2/3;                         % slope of sigmoid activation function
B = 0;                            % bias or background (sigmoid)

% [specified] fixed parameters
%--------------------------------------------------------------------------
if isfield(M,'pF')
    try, E = M.pF.E; end
    try, G = M.pF.G; end
    try, T = M.pF.T; end
    try, R = M.pF.R; end
end
 
 
% Extrinsic connections
%--------------------------------------------------------------------------
% ss = spiny stellate
% sp = superficial pyramidal
% dp = deep pyramidal
% ii = inhibitory interneurons
%--------------------------------------------------------------------------
if n > 1
    A{1} = exp(P.A{1})*E(1);      % forward  connections (sp -> ss)
    A{2} = exp(P.A{2})*E(2);      % forward  connections (sp -> dp)
    A{3} = exp(P.A{3})*E(3);      % backward connections (dp -> sp)
    A{4} = exp(P.A{4})*E(4);      % backward connections (dp -> ii)
else
    A    = {0,0,0,0};
end

% detect and reduce the strength of reciprocal (lateral) connections
%--------------------------------------------------------------------------
for i = 1:length(A)
    L    = (A{i} > exp(-8)) & (A{i}' > exp(-8));
    A{i} = A{i}./(1 + 4*L);
end

% input connections
%--------------------------------------------------------------------------
C    = exp(P.C);
 
% pre-synaptic inputs: s(V)
%--------------------------------------------------------------------------
R    = R.*exp(P.S);              % gain of activation function
F    = 1./(1 + exp(-R*xtau + B));   % firing rate
S    = F - 1/(1 + exp(B));       % deviation from baseline firing

% input
%==========================================================================
if isfield(M,'u')
    
    % endogenous input
    %----------------------------------------------------------------------
    U = u(:)*512;
    
else
    % exogenous input
    %----------------------------------------------------------------------
    U = C*u(:)*32;
    
end

 
% time constants and intrinsic connections
%==========================================================================
T    = ones(n,1)*T/1000;
G    = ones(n,1)*G;

% extrinsic connections
%--------------------------------------------------------------------------
% forward  (i)   2  sp -> ss (+ve)
% forward  (ii)  1  sp -> dp (+ve)
% backward (i)   2  dp -> sp (-ve)
% backward (ii)  1  dp -> ii (-ve)
%--------------------------------------------------------------------------
% free parameters on time constants and intrinsic connections
%--------------------------------------------------------------------------
% G(:,1)  ss -> ss (-ve self)  4
% G(:,2)  sp -> ss (-ve rec )  4
% G(:,3)  ii -> ss (-ve rec )  4
% G(:,4)  ii -> ii (-ve self)  4
% G(:,5)  ss -> ii (+ve rec )  4
% G(:,6)  dp -> ii (+ve rec )  2
% G(:,7)  sp -> sp (-ve self)  4
% G(:,8)  ss -> sp (+ve rec )  4
% G(:,9)  ii -> dp (-ve rec )  2
% G(:,10) dp -> dp (-ve self)  1
%--------------------------------------------------------------------------
% Neuronal states (deviations from baseline firing)
%--------------------------------------------------------------------------
%   S(:,1) - voltage     (spiny stellate cells)
%   S(:,2) - conductance (spiny stellate cells)
%   S(:,3) - voltage     (superficial pyramidal cells)
%   S(:,4) - conductance (superficial pyramidal cells)
%   S(:,5) - current     (inhibitory interneurons)
%   S(:,6) - conductance (inhibitory interneurons)
%   S(:,7) - voltage     (deep pyramidal cells)
%   S(:,8) - conductance (deep pyramidal cells)
%--------------------------------------------------------------------------
j     = [1 2 3 4];
for i = 1:size(P.T,2)
    T(:,j(i)) = T(:,j(i)).*exp(P.T(:,i));
end
j     = [7 2 3 4];
for i = 1:size(P.G,2)
    G(:,j(i)) = G(:,j(i)).*exp(P.G(:,i));
end

% Modulatory effects of dp depolarisation on intrinsic connection j(1)
%--------------------------------------------------------------------------
if isfield(P,'M')
    G(:,j(1)) = G(:,j(1)).*exp(-P.M*32*S(:,7));
end

 
% Motion of states: f(x)
%--------------------------------------------------------------------------
 
% Conductance
%==========================================================================
 
% Granular layer (excitatory interneurons): spiny stellate: Hidden causes
%--------------------------------------------------------------------------
u      =   A{1}*S(:,3) + U;
u      = - G(:,1).*S(:,1) - G(:,3).*S(:,5) - G(:,2).*S(:,3) + u;
f(:,2) =  (u - 2*x(:,2) - x(:,1)./T(:,1))./T(:,1);
 
% Supra-granular layer (superficial pyramidal cells): Hidden causes - error
%--------------------------------------------------------------------------
u      = - A{3}*S(:,7);
u      =   G(:,8).*S(:,1) - G(:,7).*S(:,3) + u;
f(:,4) =  (u - 2*x(:,4) - x(:,3)./T(:,2))./T(:,2);
 
% Supra-granular layer (inhibitory interneurons): Hidden states - error
%--------------------------------------------------------------------------
u      = - A{4}*S(:,7);
u      =   G(:,5).*S(:,1) + G(:,6).*S(:,7) - G(:,4).*S(:,5) + u;
f(:,6) =  (u - 2*x(:,6) - x(:,5)./T(:,3))./T(:,3);
 
% Infra-granular layer (deep pyramidal cells): Hidden states
%--------------------------------------------------------------------------
u      =   A{2}*S(:,3);
u      = - G(:,10).*S(:,7) - G(:,9).*S(:,5) + u;
f(:,8) =  (u - 2*x(:,8) - x(:,7)./T(:,4))./T(:,4);
 
% Voltage
%==========================================================================
f(:,1) = x(:,2);
f(:,3) = x(:,4);
f(:,5) = x(:,6);
f(:,7) = x(:,8); 
 
if nargout < 2; return, end

% Jacobian and delay matrix (will be properly computed in
% tnudcm_compute_xtau)
%==========================================================================
J  = 0; 
D  = [];


