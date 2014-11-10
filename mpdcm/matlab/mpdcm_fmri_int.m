function [y] = mpdcm_fmri_int(u, theta, ptheta, sloppy)
%% 
% sloppy -- Don't check the input
% tflag -- Test flag, this is done 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 4
    sloppy = 0;
end


if ~sloppy
    assert(mpdcm_num_devices() > 0, 'mpdcm:fmri:int:no_gpu', ...
        'No GPU device available.');
    mpdcm_fmri_int_check_input(u, theta, ptheta);  
end

for i = 1:numel(theta)
    theta{i}.C = theta{i}.C/16;
    [k1, k2, k3] = mpdcm_fmri_k(theta{i});
    theta{i}.k1 = k1;
    theta{i}.k2 = k2;
    theta{i}.k3 = k3;
end

y = c_mpdcm_fmri_int(u, theta, ptheta);

% Downsample

for i = 1:numel(y)
    y{i} = y{i}(:, 1:2:end)';
end


end
