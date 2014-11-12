function test_tapas_vblm(fp) 
%% Test the function tapas_vblm
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 1
    fp = 1;
end 

fprintf(fp, '================\n Test %s\n================\n', 'tapas_vblm');

test_polynomial_regression(fp);


end

function test_polynomial_regression(fp)
%% Test regression using simple polynomial regression problm

    [x, y, ny] = polynomial_regression_data();


    % Test whether there is any clear bug
    try
        [q, stats, q_trace] = tapas_vblm(ny, x);
        fprintf(fp, '       Passed\n');
    catch err
        fprintf(fp, '   Not passed at line %d\n', err.stack(1).line);
    end

    % Test the parameter recovery
    try
        [q, stats, q_trace] = tapas_vblm(ny, x);
        fprintf(fp, '       Passed\n');
    catch err
        fprintf(fp, '   Not passed at line %d\n', err.stack(1).line);
    end


end

function [x, y, ny] = polynomial_regression_data()
%% Returns a predefined data set

    NP = 300;

    rng(329840);

    x = bsxfun(@power, kron(ones(1, 5), linspace(-1, 1, NP)'), 1:5);
    y = x * [-0.3, 0.1, 0.9, -1.1, 0.8]';
    ny = y + 0.8 * randn(NP, 1);

end
