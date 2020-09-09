function [response, x, s, f, v, q] = tapas_huge_bold( A, B, C, D, tau, ...
    kappa, epsilon, R, u, L, E_0, r_0, V_0, vartheta_0, alpha, gamma, TR, ...
    TE, dt)
% Integrates the DCM forward equations to generate the predicted fMRI bold
% time series.
% 
% INPUTS:
%   A, B, C, D - DCM connectivity matrices.
%   tau        - Venous transit time.
%   kappa      - Decay of vasodilatory signal.
%   epsilon    - Ratio of intra- and extravascular signal.
%   R          - Number of regions.
%   u          - Experimental stimuli.
%   L          - Number of experimental stimuli.
%   E_0        - Resting oxygen extraction fraction.
%   r_0        - Slope of intravascular relaxation rate.
%   V_0        - Resting venous volume.
%   vartheta_0 - Frequency offset at the outer surface of magnetized
%                vessels (Hz). 
%   alpha      - Grubb's exponent.
%   gamma      - rate constant of feedback regulation.
%   TR         - Repetition time.
%   TE         - Echo time.
%   dt         - Sampling interval of inputs.
% 
% OUTPUTS:
%   response - matrix of predicted response for each region
%                  (column-wise) 
%   x        - time series of neuronal states
%   s        - time series of vasodilatory signal 
%   f1       - time series of flow
%   v1       - time series of blood volume
%   q1       - time series of deoxyhemoglobin content.
% 

% 
% REFERENCE:
%   Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A.
%   Robinson, Karl J. Friston (2007). Comparing hemodynamic models with
%   DCM. NeuroImage, 38: 387-401
% 
% https://doi.org/10.1016/j.neuroimage.2007.07.040
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch), Sudhir Shankar Raman
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


nt = size(u, 1);
rSmp = TR/dt;
C = C'/rSmp;
if isempty(D)
    D = zeros(R, R, R);
end

tau     = 2.*exp(tau);
kappa   = .64.*exp(kappa);
epsilon = exp(epsilon);

% resting oxygen extraction fraction
E_0 = repmat(E_0, 1, R);

k1  = 4.3*vartheta_0*TE*E_0;
k2  = epsilon.*(r_0*E_0*TE);
k3  = 1 - epsilon;



% Integrate the dynamical system
[x,s,f,v,q]  = tapas_huge_int_euler(...
    A',...
    full(u*C),...
    full(u),...
    permute(B,[2 1 3]),...
    permute(D,[2 1 3]),...
    E_0,...
    1/alpha,...
    tau,...
    gamma,...
    kappa,...
    [dt,nt,R,L,0,any(B(:)),any(D(:))]);


% generate the BOLD response
response = V_0*( ...
    bsxfun(@times,k1,...
        (1 - (q(rSmp:rSmp:end,:)))) +...
    bsxfun(@times,k2,...
        (1 - (q(rSmp:rSmp:end,:)./...
        v(rSmp:rSmp:end,:)))) +...
    bsxfun(@times,k3,...
        (1-v(rSmp:rSmp:end,:))));

% demean response
response = bsxfun(@minus, response, mean(response, 1));

end