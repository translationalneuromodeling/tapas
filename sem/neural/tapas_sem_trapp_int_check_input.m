function tapas_sem_trapp_int_check_input(u, theta, ptheta)
% Check whether the input is compatible with mpdcm. If not an error is 
% returned.
% 
% Input:
% u         -- Structure. Inputs to DCM in mpdcm format.
% theta     -- Structure. Model parameters in mpdcm format.
% ptheta    -- Structure. Priors of the model in mpdcm format.
%
% Output:
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2016 by Eduardo Aponte <aponteeduardo@gmail.com>
%

tapas_mpdcm_check_input_u(u);
tapas_mpdcm_check_input_ptheta(ptheta);

check_theta(theta)

su = size(u);
st = size(theta);

assert(su(1) == st(1), ...
    'mpdcm:fmri:int:input:dmatch', ...
    'Dimensions of u and theta doesn''t match')

su = size(u{1});
du = theta{1}.dim_u;
assert(su(1) == du, ...
    'mpdcm:fmri:int:input:theta:dim_u:match_u', ...
    'theta.dim_u doesn''t match u.');
end


function check_theta(theta)
%% Throws an error if something is wrong with theta

assert(iscell(theta), ...
    'mpdcm:fmri:int:input:theta:not_cell', ...
    'theta should be a cell array')
assert(ndims(theta) == 2, ...
    'mpdcm:fmri:int:input:theta:ndim', ...
    'theta should be two dimensional, number of dimensions is %d', ...
    ndims(theta))

ofD = 0;
nfD = 0;

for i = 1:numel(theta)

    assert(isstruct(theta{i}), ...
        'mpdcm:fmri:int:input:theta:cell:not_struct', ...
        'theta{%d} is not struct', i)

    assert(isscalar(theta{i}), ...
        'mpdcm:fmri:int:input:theta:cell:ndim', ...
        'theta{%d} should be 1 x 1', i)

    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'dim_x', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'dim_u', i);

    if i == 1
        dx = theta{i}.dim_x;
        du = theta{i}.dim_u;
    end

    assert(dx == theta{i}.dim_x, ...
        'mpdcm:fmri:int:input:theta:cell:dim_x:not_match', ...
        'theta{%d}.dim_x doesn''t match previous values', i);

    assert(du == theta{i}.dim_u, ...
        'mpdcm:fmri:int:input:theta:cell:dim_u:not_match', ...
        'theta{%d}.dim_u doesn''t match previous values', i);

    tapas_mpdcm_check_input_matrix(theta, [dx, dx], 'A', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, dx], 'B', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, du], 'C', i);

    tapas_mpdcm_check_input_matrix(theta, [dx, 1], 'x0', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, 1], 'beta', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, 1], 'theta', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'tau', i);

end

end


