function [y, x] = tapas_dcm_euler_gen(DCM, Ep)
% [y, x] = tapas_dcm_euler_gen(DCM, Ep)
% 
% Generates synthetic fMRI data under a given signal to noise ratio (SNR) 
% with the fixed hemodynamic convolution kernel
% 
%   Input:
%   	DCM         - model structure
%       Ep          - data-generating parameters
%
%   Output:
%       y           - generated BOLD signal
%       x           - generated neuronal signal
%

% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2018 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% number of regions
nr = size(Ep.A,1);
A = Ep.A';

% Transpose each nrxnr matrix
mArrayB = permute(Ep.B, [2 1 3]);

% create an empty D-matrix
if (sum(Ep.D(:))==0)
    Ep.D = zeros(nr,nr,nr);
end

% Transpose each nrxnr matrix
mArrayD = permute(Ep.D, [2 1 3]);
if (isempty(mArrayD))
    mArrayD = zeros(nr,nr,nr);
end

% driving inputs
C = DCM.U.u*(Ep.C'/16);
U = full(DCM.U.u);

% hemodynamic constants
H = [0.64 0.32 2.00 0.32 0.32];

% constants for hemodynamic model
oxygenExtractionFraction = 0.32*ones(nr, 1);
alphainv                 = 1/H(4);
tau                      = H(3)*exp(Ep.transit);
gamma                    = H(2);
kappa                    = H(1)*exp(Ep.decay);
epsilon                  = 1.0*exp(Ep.epsilon)*ones(nr, 1);

% parameter list
paramList = [DCM.U.dt size(U,1) nr size(U,2) 0 1 1];

% neuronal signal and time courses for hemodynamic parameters
[x,~,~,v,q] = dcm_euler_integration(A,C,U,mArrayB,mArrayD,...
                   oxygenExtractionFraction,alphainv,tau,gamma,kappa,paramList);

% constants for BOLD signal equation
relaxationRateSlope      = 25;
frequencyOffset          = 40.3;  
oxygenExtractionFraction = 0.4*ones(1,nr);
echoTime                 = 0.04;
restingVenousVolume      = 4;

% coefficients of BOLD signal equation
coefficientK1  = 4.3*frequencyOffset*echoTime*oxygenExtractionFraction;
coefficientK2  = epsilon'.*(relaxationRateSlope*oxygenExtractionFraction*echoTime);
coefficientK3  = 1 - epsilon';

% get the Euler indices
Indices = DCM.M.idx;

% BOLD signal time course
y = restingVenousVolume*( bsxfun(@times,coefficientK1,(1 - (q(Indices,:)))) +...
            bsxfun(@times,coefficientK2,(1 - (q(Indices,:)./v(Indices,:)))) +...
            bsxfun(@times,coefficientK3,(1-v(Indices,:))));        

end
