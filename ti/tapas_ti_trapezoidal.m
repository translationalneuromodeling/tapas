function [fe] = tapas_ti_trapezoidal(llh, T)
%% Computes the free energy using the trapezoidal rule
%
% Input
%       llh     -- Samples from the log likelihod
%       T       -- Temperatures at which the samples were drawn
% Output
%       fe      -- Free energy
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ellh = mean(llh, 1);
fe = trapz(T, llh);

end

