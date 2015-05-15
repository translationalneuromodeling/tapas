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

check_u(u)
check_theta(theta)
check_ptheta(ptheta)

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

function check_u(u)
%% Throws an error if something is wrong with theta

assert(iscell(u), ...
    'mpdcm:fmri:int:input:u:not_cell', ...
    'u should be a cell array')

assert(ndims(u) == 2, ...
    'mpdcm:fmri:int:input:u:ndim', ...
    'u should be two dimensional, number of dimensions is %d', ndims(u))

su = size(u);

assert(su(2) == 1, ...
    'mpdcm:fmri:int:input:u:dsize', ...
    'Second dimension of u should have size 1, size is %d', su(2))

for i = 1:numel(u)
    assert(isnumeric(u{i}), ...
        'mpdcm:fmri:int:input:u:cell:not_numeric', ...
        'u{%d} should be numeric.', i);
    assert(ndims(u{i}) == 2, ...
        'mpdcm:fmri:int:input:u:cell:not_real', ...
        'u{%d} should be two dimensional.', i);
    assert(isreal(u{i}), ...
        'mpdcm:fmri:int:input:u:cell:not_real', ...
        'u{%d} should be real.', i);
    assert(~issparse(u{i}), ...
        'mpdcm:fmri:int:input:u:cell:sparse', ...
        'u{%d} should not be sparse', i);

    if i == 1
        os = size(u{i});
    end

    assert(all(os == size(u{i})), ...
        'mpdcm:fmri:int:input:u:cell:not_match', ...
        'All cells of u should have the same dimensions');
    os = size(u{i});
end

end

function check_matrix(theta, ns, field, element)
%% Checks a matrix in theta.
%
% Input
% theta -- Cell array
% ns -- Expected size of the matrix
% field -- String with the respetive field
% Element -- Index

    assert(isfield(theta{element}, field), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:missing', field), ...
        'Element theta{%d} doesn''t have field %s', element, field);
    m = getfield(theta{element}, field);
    assert(isnumeric(m), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:not_numeric', field), ...
        'theta{%d}.%s should be numeric', element, field);
    assert(isreal(m), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:not_real', field), ...
        'theta{%d}.%s should be real', element, field);
    assert(~issparse(m), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:sparse', field), ...
        'theta{%d}.%s should not be sparse', element, field);
    assert(ndims(m) == numel(ns), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:ndim', field), ...
        'theta{%d}.%s should have %d dimensions', ...
        element, field, numel(ns))
    assert(all(size(m) == ns), ...
        sprintf('mpdcm:fmri:int:input:theta:cell:%s:dsize', field), ...
        'theta{%d}.%s should have dimensions [%d,%d], instead [%d,%d].', ...
        element, field, ns(1), ns(2), size(m, 1), size(m, 2));

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

    check_matrix(theta, [1, 1], 'dim_x', i);
    check_matrix(theta, [1, 1], 'dim_u', i);

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

    check_matrix(theta, [1, 1], 'fA', i);
    check_matrix(theta, [1, 1], 'fB', i);
    check_matrix(theta, [1, 1], 'fC', i);
    check_matrix(theta, [1, 1], 'fD', i);


    % Check if the flag is the same for all of the data structures.

    if i == 1
        ofD = theta{i}.fD;
    end

    assert(ofD == theta{i}.fD, ...
        'mpdcm:fmri:int:input:theta:cell:fD:not_match', ...
        'theta{%d}.fD doesn''t match previous values', i);

    % Check matrices

    check_matrix(theta, [dx, dx], 'A', i);
    check_matrix(theta, [dx, dx, du], 'B', i);
    check_matrix(theta, [dx, du], 'C', i);
    if theta{i}.fD 
        check_matrix(theta, [dx, dx, dx], 'D', i);
    end

    check_matrix(theta, [dx, 1], 'K', i);
    check_matrix(theta, [dx, 1], 'tau', i);
    check_matrix(theta, [1, 1], 'V0', i);
    check_matrix(theta, [1, 1], 'E0', i);

    % In one of the equations, the expression (1 - E0)**(1/f) shows up. This
    % is problematic if E0 > 1

    assert(all(theta{i}.E0 < 1), ...
        'mpdcm:fmri:int:input:theta:E0:numeric_error', ...
        'theta{%d}.E0 is larger than 1. This would cause numerical errors.',...
        i);

    check_matrix(theta, [1, 1], 'k1', i);
    check_matrix(theta, [1, 1], 'k2', i);
    check_matrix(theta, [1, 1], 'k3', i);
    check_matrix(theta, [1, 1], 'alpha', i);
    check_matrix(theta, [1, 1], 'gamma', i);

end
end

function check_ptheta_scalar(ptheta, field)
%% Checks for scalar values in ptheta

assert(isfield(ptheta, field), ...
    sprintf('mpdcm:fmri:int:input:ptheta:%s:missing', field), ...
    'ptheta should have field %s', field);

ascalar = getfield(ptheta, field);

assert(isscalar(ascalar), ...
    sprintf('mpdcm:fmri:int:input:ptheta:%s:not_scalar', field), ...
    'ptheta.%s should be scalar', field);

assert(isnumeric(ascalar), ...
    sprintf('mpdcm:fmri:int:input:ptheta:%s:not_numeric', field), ...
    'ptheta.%s should be numeric', field);
assert(isreal(ascalar), ...
    sprintf('mpdcm:fmri:int:input:ptheta:%s:not_real', field), ...
    'ptheta.%s should be real', field);
assert(~issparse(ascalar), ...
    sprintf('mpdcm:fmri:int:input:theta:cell:%s:sparse', field), ...
    'ptheta.%s should not be sparse', field);

end

function check_ptheta(ptheta)
%% Throws an error if something is wrong with ptheta

assert(isstruct(ptheta), ...
    'mpdcm:fmri:int:input:ptheta:not_struct', ...         
    'ptheta should be a struct')

check_ptheta_scalar(ptheta, 'dt');

dt = getfield(ptheta, 'dt');

assert(0 < dt && dt <= 1, ...
    sprintf('mpdcm:fmri:int:input:theta:cell:%s:val', 'dt'), ...
    'ptheta.%s should not be < 0 and > 1', 'dt');

check_ptheta_scalar(ptheta, 'dyu');
dyu = getfield(ptheta, 'dyu');

assert(0 < dyu && dyu <= 1, ...
    sprintf('mpdcm:fmri:int:input:theta:cell:%s:val', 'dyu'), ...
    'ptheta.%s should not be < 0 and > 1', 'dyu');

end
