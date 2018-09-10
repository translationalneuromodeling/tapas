function [npp, ipp, nt] = tapas_genpath(t, llh, minv, type)
%% Create a new path for thermodynamic integration using a Hermite 
% polynomial.
%
% Input
%   t       -- Temperature schedule. Nx1 double array
%   llh     -- llh. Samples of the temperature. N x M array, where M is the
%               number of samples.
%   minv    -- Number of the lowest element that can be used for fitting.
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

n = 4;
if nargin < 4
    type = 'hermite';
end

vllh = var(llh')';
mllh = mean(llh, 2);

mllh = sgolayfilt(mllh, 1, 5);

% Use a Hermite polynomial
npp = ppcreate(t(minv:end), mllh(minv:end), type, vllh(minv:end));

% Create the inverse function
ipp = ppcreate(npp, 'inv');

nv = numel(mllh) - minv + 1;

% First create an array of values
v = linspace(mllh(minv), mllh(end), nv);

% Create the new temperature schedule
nt = zeros(nv, 1);
for i = 1:nv
    tx = v(i);
    ty = ipp(tx);
    % Always appoximate to one.
    if i == nv
        ty = 1.0;
    end
    nt(i) = ty;
end

end
