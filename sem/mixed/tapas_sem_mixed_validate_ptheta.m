function [ptheta] = tapas_sem_mixed_validate_ptheta(ptheta)
%% Make sute that the input is consistent.
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

if ~isfield(ptheta, 'x')
    error('tapas:sem:mixed:ptheta', 'Design matrix ptheta.m is not specified');
end

if ~isfield(ptheta, 'jm')
    error('tapas:sem:mixed:ptheta', ...
        'Reduction matrix ptheta.jm is not specified');
end

if ~isfield(ptheta, 'mixed')
    error('tapas:sem:mixed:ptheta', ...
        'Random effects ptheta.mixed is not specified');
end

% Now make sure that the every thing is consistent

jm1, jm2 = size(ptheta.jm);
x1, x2 = size(ptheta.x);
m1, m2 = size(ptheta.mixed);

test_matrix(ptheta.jm, jm1, jm2, 'jm');
test_matrix(ptheta.x, jm2, x2, 'x');
test_matrix(ptheta.mixed, jm2, m2, 'mixed');

ptheta.x = logical(ptheta.x);

end


function test_matrix(amatrix, d1, d2, name)
% Make standard test on a matrix

if ~isnumeric(amatrix)
    error(sprintf('tapas:sem:mixed:ptheta:%s', name), ...
        '%s not numeric', name);

end

if ~all(size(amatrix) == [d1, d2])
    error(sprintf('tapas:sem:mixed:ptheta:%s', name), ...
        '%s dimension are %d, %d, instead %d, %d', ...
        name, d1, d2, size(amatrix, 1), size(amatrix, 2));
end


end
