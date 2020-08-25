function [f,J,D] = tapas_ceode_fx_erp(x,u,P,M, xtau)
% [f,J,D] = tapas_ceode_fx_erp(x, u, P, M, xtau)
% 
% Computation of the dynamical (state) equation for a neural mass model
% (ERP model). Compatibile with continuous extension for ODE
% methods. Adapted from spm_fx_erp.m (original function license below).
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
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging
%
% Karl Friston
% $Id: spm_fx_erp.m 6720 2016-02-15 21:06:55Z karl $
%--------------------------------------------------------------------------

x  = spm_unvec(x,M.x);      % neuronal states
n  = size(x,1);             % number of sources


if nargin < 5
    xtau = x; 
    xtau = spm_unvec(xtau, M.x);
end

% [default] fixed parameters
%--------------------------------------------------------------------------
E = [1 1/2 1/8]*32;         % extrinsic rates (forward, backward, lateral)
G = [1 4/5 1/4 1/4]*128;    % intrinsic rates (g1 g2 g3 g4)
D = [2 16];                 % propogation delays (intrinsic, extrinsic)
H = [4 32];                 % receptor densities (excitatory, inhibitory)
T = [8 16];                 % synaptic constants (excitatory, inhibitory)
R = [2 1]/3;                % parameters of static nonlinearity

% [specified] fixed parameters
%--------------------------------------------------------------------------
if isfield(M,'pF')
    try, E = M.pF.E; end
    try, G = M.pF.H; end
    try, D = M.pF.D; end
    try, H = M.pF.G; end
    try, T = M.pF.T; end
    try, R = M.pF.R; end
end

% test for free parameters on intrinsic connections
%--------------------------------------------------------------------------
try
    G = G.*exp(P.H);
end
G     = ones(n,1)*G;

% exponential transform to ensure positivity constraints
%--------------------------------------------------------------------------
if n > 1
    A{1} = exp(P.A{1})*E(1);
    A{2} = exp(P.A{2})*E(2);
    A{3} = exp(P.A{3})*E(3);
else
    A = {0,0,0};
end
C     = exp(P.C);

% intrinsic connectivity and parameters
%--------------------------------------------------------------------------
Te    = T(1)/1000*exp(P.T(:,1));         % excitatory time constants
Ti    = T(2)/1000*exp(P.T(:,2));         % inhibitory time constants
He    = H(1)*exp(P.G(:,1));              % excitatory receptor density
Hi    = H(2)*exp(P.G(:,2));              % inhibitory receptor density

% pre-synaptic inputs: s(V)
%--------------------------------------------------------------------------
R     = R.*exp(P.S);
S     = 1./(1 + exp(-R(1)*(xtau - R(2)))) - 1./(1 + exp(R(1)*R(2)));

% input
%==========================================================================
if isfield(M,'u')
    
    % endogenous input
    %----------------------------------------------------------------------
    U = u(:)*64;
    
else
    % exogenous input
    %----------------------------------------------------------------------
    U = C*u(:)*2;
end


% State: f(x)
%==========================================================================

% Supragranular layer (inhibitory interneurons): Voltage & depolarizing current
%--------------------------------------------------------------------------
f(:,7) = x(:,8);
f(:,8) = (He.*((A{2} + A{3})*S(:,9) + G(:,3).*S(:,9)) - 2*x(:,8) - x(:,7)./Te)./Te;

% Granular layer (spiny stellate cells): Voltage & depolarizing current
%--------------------------------------------------------------------------
f(:,1) = x(:,4);
f(:,4) = (He.*((A{1} + A{3})*S(:,9) + G(:,1).*S(:,9) + U) - 2*x(:,4) - x(:,1)./Te)./Te;

% Infra-granular layer (pyramidal cells): depolarizing current
%--------------------------------------------------------------------------
f(:,2) = x(:,5);
f(:,5) = (He.*((A{2} + A{3})*S(:,9) + G(:,2).*S(:,1)) - 2*x(:,5) - x(:,2)./Te)./Te;

% Infra-granular layer (pyramidal cells): hyperpolarizing current
%--------------------------------------------------------------------------
f(:,3) = x(:,6);
f(:,6) = (Hi.*G(:,4).*S(:,7) - 2*x(:,6) - x(:,3)./Ti)./Ti;

% Infra-granular layer (pyramidal cells): Voltage
%--------------------------------------------------------------------------
f(:,9) = x(:,5) - x(:,6);

if nargout < 2; return, end

% Jacobian and delay matrix (will be properly computed in
% tnudcm_compute_xtau)
%==========================================================================
J  = 0; 
D  = [];





