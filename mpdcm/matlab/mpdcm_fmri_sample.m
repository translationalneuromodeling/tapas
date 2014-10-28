function [np] = mpdcm_fmri_sample(op, ptheta, htheta, v)
%% Draws a new sample from a Gaussian proposal distribution.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 4
    v = cell(size(op));
    v(:) = {1};
end

nt = numel(op);
np = cell(size(op));

for i = 1:nt
    np{i} = full(op{i} + htheta.c_c * v{i} * randn(size(op{i})));
end

end
