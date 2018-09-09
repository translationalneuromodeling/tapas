function tapas_mpdcm_erp_int_check_input_host(u, theta, ptheta)
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
    'mpdcm:erp:int:input:dmatch', ...
    'Dimensions of u and theta doesn''t match')

su = size(u{1});
du = theta{1}.dim_u;
assert(su(1) == du, ...
    'mpdcm:erp:int:input:theta:dim_u:match_u', ...
    'theta.dim_u doesn''t match u.');
end


function check_theta(theta)
%% Throws an error if something is wrong with theta

fscalars = {};
farrays = {'dim_x', 'dim_u', 'Au', 'gamma1', 'gamma2', 'gamma3', 'gamma4', ...
    'r1', 'r2', 'er1r2', 'tau_e2', 'tau_es2','tau_i2', 'tau_is2', 'A13', 'A23'};


assert(iscell(theta), ...
    'mpdcm:erp:int:input:theta:not_cell', ...
    'theta should be a cell array')
assert(ndims(theta) == 2, ...
    'mpdcm:erp:int:input:theta:ndim', ...
    'theta should be two dimensional, number of dimensions is %d', ...
    ndims(theta))

odim_x = nan;
odim_y = nan;

for i = 1:numel(theta)

    assert(isstruct(theta{i}), ...
        'mpdcm:erp:int:input:theta:cell:not_struct', ...
        'theta{%d} is not struct', i)

    assert(isscalar(theta{i}), ...
        'mpdcm:erp:int:input:theta:cell:ndim', ...
        'theta{%d} is not struct', i)
       

    for j = 1:numel(fscalars)
        tapas_mpdcm_check_input_matrix(theta, ...
            [1, 1], fscalars{j}, 1);
    end

    dim_x = theta{i}.dim_x;
    dim_u = theta{i}.dim_u;

    assert(isnumeric(dim_x) && round(dim_x) == dim_x, 'dim_x should be integer')
    assert(isnumeric(dim_u) && round(dim_u) == dim_u, 'dim_u should be integer')

    if isnan(odim_x)
        odim_x = dim_x;
        odim_u = dim_u;
    end

    assert(odim_x == dim_x, 'dim_x should be constant');
    assert(odim_u == dim_u, 'dim_u should be constant');
 
    for j = 1:numel(farrays)
        tapas_mpdcm_check_input_matrix(theta, ...
            [dim_x, 1], farrays{j}, 1);
    end
end


end


