function tapas_linear_validate_data(y, u)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~iscell(y)
    error('tapas:linear:validate_data', 'y should be a cell array');
end

if ~iscell(u)
    error('tapas:linear:validate_data', 'y should be a cell array');
end

if ~ (size(y, 2) == 1)
     error('tapas:linear:validate_data', 'y should be Nx1');
end   

if ~ (size(u, 2) == 1)
     error('tapas:linear:validate_data', 'u should be Nx1');
end   


if ~ (size(y, 1) == size(u, 1))
    error('tapas:linear:validate_data', 'y and u should have the same size');
end

np = size(y, 1);

[nt, d2] = size(y{1});

if d2 ~= 1
    error('tapas:linear:validate_data', 'y{1} should be Nx1');
end

[nu, nb] = size(u{1});

if nu ~= nt
    error('tapas:linear:validate_data', ...
        'u{1} first dimension should be %d', nt);
end

for i = 1:np
    if ~ (size(y{i}, 2) == 1)
        error('tapas:linear:validate_data', ...
            'y{%d} first dimension should be 1', i);
    end

    if ~ (size(u{i}, 1) == size(y{i}, 1))
         error('tapas:linear:validate_data', ...
            'y{%d} and u{%d} first dimension should be equal', i, i);     
    end

    if ~(size(u{i}, 2) == nb)
        error('tapas:linear:validate_data',  ...
            'u{%d} second dimension is inconsistent', i);
    end
end

end
