function [np] = mpdcm_fmri_sample(op, ptheta, htheta)
%% Draws a new sample from a Gaussian proposal distribution.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

nt = numel(op);
np = cell(size(op));

for i = 1:nt
    np{i} = full(op{i} + htheta.c_c * randn(size(op{i})));
end

end
