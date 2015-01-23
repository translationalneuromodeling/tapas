function [y] = mpdcm_fmri_int(u, theta, ptheta, sloppy)
%% Integrates the system of differential equations specified by the input.
%
% Input:
% u -- Cell array of experimental input
% theta -- Cell array of model parameteres
% ptheta -- Model priors or constants
% sloppy -- If true don't check the input. 
%
% Ouput:
% y -- Cell array of predicted signals.
%
% If the input is not compliant, it's very likely that a segmentation fault
% happens and that matlab closes. It should only be used once the input has
% been check at least once and changes to them are done via well tested
% functions.
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

if isfield(ptheta, 'integ')
    switch ptheta.integ
    case 'euler'
        integ = @(u, theta, ptheta) c_mpdcm_fmri_euler(u, theta, ptheta);
    case 'kr4'
        integ = @(u, theta, ptheta) c_mpdcm_fmri_kr4(u, theta, ptheta);
    otherwise 
        error('mpdcm:fmri:int:input', ... 
            'Unknown method for ptheta.int');
    end
else
    integ = @(u, theta, ptheta) c_mpdcm_fmri_euler(u, theta, ptheta); 
end

hps = mpdcm_fmri_get_hempars();

for i = 1:numel(theta)
    theta{i}.C = theta{i}.C/16;
    [k1, k2, k3] = mpdcm_fmri_k(theta{i});
    theta{i}.k1 = k1;
    theta{i}.k2 = k2;
    theta{i}.k3 = k3;
    % Change the parametrization
    theta{i}.K = hps.K * exp(theta{i}.K);
    theta{i}.tau = hps.tau + theta{i}.tau;
end

% Integrate

y = integ(u, theta, ptheta);

% Downsample

for i = 1:numel(y)
    y{i} = y{i}(:, 2:2:end)';
    if isfield(theta{i}, 'ny')
        y{i} = y{i}(1:theta{i}.ny, :);
    end
end


end
