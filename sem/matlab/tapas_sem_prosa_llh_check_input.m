function tapas_sem_prosa_llh_check_input(t, a, u, theta) 
%% Checks the input to the likelihood function 
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%

DIM_THETA = tapas_sem_prosa_ndims();

ns = size(t, 1);
np = size(theta, 2);

check_matrix(t, [ns, 1], 't');
check_matrix(a, [ns, 1], 'a');
check_matrix(u, [ns, 1], 'u');

if np > 1
    assert(np == ns, sprintf('tapas:sem:llh:input:size'), ...
        'Number fo parameters should be %d, instead %d', ns, np);
end

check_matrix(theta, [2 * DIM_THETA, np], 'theta'); 
%check_delays(t, a, u, theta);

end

function check_matrix(mat, ns, field)
%% Checks a matrix in theta.
%
% Input
%   mat -- Matrix
%   ns -- Expected size of the matrix
%   field -- String with the name of the parameter

assert(isnumeric(mat), ...
    sprintf('tapas:sem:llh:input:%s:not_numeric', field), ...
    '%s should be numeric', field);
assert(isreal(mat), ...
    sprintf('tapas:sem:llh:input:%s:not_real', field), ...
    '%s should be real', field);
assert(~issparse(mat), ...
    sprintf('tapas:sem:llh:input:%s:sparse', field), ...
    '%s should not be sparse', field);
assert(ndims(mat) == 2, ...
    sprintf('tapas:sem:llh:input:%s:ndim', field), ...
    '%s should have 2 dimensions', field)
assert(all(size(mat) == ns), ...
    sprintf('tapas:sem:llh:input:%s:dsize', field), ...
    '%s should have dimensions [%d,%d], instead [%d,%d].', ...
    field, ns(1), ns(2), size(mat, 1), size(mat, 2));
%assert(all(mat(:) >= 0 ), ...
%    sprintf('tapas:sem:llh:input:%s:range', field), ...
%    '%s should be positive', field)
end


function check_delays(t, a, u, theta)
% Check if the delays make a trial implaussible 

[PROSACCADE, ANTISACCADE] = tapas_sem_action_codes();

assert(all(t > theta(13)), 'tapas:sem:llh:input:main_delay', ...
    'Main delay %0.5f is higher than at least one saccade', theta(13));

assert(all(t(a == ANTISACCADE & u == PROSACCADE) > theta(13) + theta(14)),...
    'tapas:sem:llh:input:pro_delay', ...
    ['Delay of antisaccade in a prosaccade trial %0.5f is higher than at '...
    'least one saccade'], theta(13) + theta(14));

assert(all(t(a == ANTISACCADE & u == PROSACCADE) > theta(13) + theta(15)),...
    'tapas:sem:llh:input:anti_delay', ...
    ['Delay of antisaccade in an antisaccade trial %0.5f is higher than at ' ...
   'least one saccade'], theta(13) + theta(15));

end

