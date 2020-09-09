function [y, pst] = tapas_ceode_gen_erp(P, M, U)
% [y, pst] = tapas_ceode_gen_erp(P, M, U)
% 
% Compute predicted data based on a specified DCM, with variable integrator
% specification. 
%
% Adapted from spm_gen_erp.m (original function lincense below).
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
%
% Karl Friston
% $Id: spm_gen_erp.m 5758 2013-11-20 21:04:01Z karl $
%--------------------------------------------------------------------------


% default inputs - one trial (no between-trial effects)
%--------------------------------------------------------------------------
if nargin < 3, U.X = sparse(1,0); end


% peristimulus time
%--------------------------------------------------------------------------
if nargout > 1
    pst = (1:M.ns)*U.dt - M.ons/1000;
end

% within-trial (exogenous) inputs
%==========================================================================
if ~isfield(U,'u')
    
    % peri-stimulus time inputs
    %----------------------------------------------------------------------
    U.u = feval(M.fu,(1:M.ns)*U.dt,P,M);
    
end

if isfield(M,'u')
    
    % remove M.u to preclude endogenous input
    %----------------------------------------------------------------------
    M = rmfield(M,'u');
    
end

% between-trial (experimental) inputs
%==========================================================================
if isfield(U,'X')
    X = U.X;
else
    X = sparse(1,0);
end

if ~size(X,1)
    X = sparse(1,0);
end

% cycle over trials
%==========================================================================
y      = cell(size(X,1),1);
for  c = 1:size(X,1)
    
    % condition-specific parameters
    %----------------------------------------------------------------------
    Q = spm_gen_Q(P, X(c, :));
    
    % solve for steady-state - for each condition
    %----------------------------------------------------------------------
    M.x  = spm_dcm_neural_x(Q,M);
    
    % integrate DCM - for this condition
    %----------------------------------------------------------------------
    y{c} = feval(M.int, Q, M, U);
        
end





