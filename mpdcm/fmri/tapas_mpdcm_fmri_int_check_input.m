function tapas_mpdcm_fmri_int_check_input(u, theta, ptheta)
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
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
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
        'theta{%d} is not struct', i)

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

    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'fA', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'fB', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'fC', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'fD', i);


    % Check if the flag is the same for all of the data structures.

    if i == 1
        ofD = theta{i}.fD;
    end

    assert(ofD == theta{i}.fD, ...
        'mpdcm:fmri:int:input:theta:cell:fD:not_match', ...
        'theta{%d}.fD doesn''t match previous values', i);

    % Check matrices

    tapas_mpdcm_check_input_matrix(theta, [dx, dx], 'A', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, dx, du], 'B', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, du], 'C', i);

    if theta{i}.fD 
        tapas_mpdcm_check_input_matrix(theta, [dx, dx, dx], 'D', i);
    end

    tapas_mpdcm_check_input_matrix(theta, [dx, 1], 'K', i);
    tapas_mpdcm_check_input_matrix(theta, [dx, 1], 'tau', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'V0', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'E0', i);

    % In one of the equations, the expression (1 - E0)**(1/f) shows up. This
    % is problematic if E0 > 1

    assert(all(theta{i}.E0 < 1), ...
        'mpdcm:fmri:int:input:theta:E0:numeric_error', ...
        'theta{%d}.E0 is larger than 1. This would cause numerical errors.',...
        i);

    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'alpha', i);
    tapas_mpdcm_check_input_matrix(theta, [1, 1], 'gamma', i);

end
end


