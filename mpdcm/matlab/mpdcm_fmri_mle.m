function [mX, cX] = mpdcm_fmri_mle(y, u, theta, ptheta)
%% Computes a local maximum of the parameters by minimizing the square error.
%
% It uses regularized Gaussian-Newton optimization with GL 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

op = mpdcm_fmri_get_parameters(theta, ptheta);
op = op{1};

np = zeros(size(op));

dt = 1e-3;

mpdcm_fmri_int_check_input(u, theta, ptheta);

for j = 1:100

    if mod(j, 10) == 0
        fprintf(1, 'Iteration: %d, RSE: %0.5f\n', j, e'*e);
    end
    
    [dfdx, ny] = mpdcm_fmri_gradient(op, u, theta, ptheta, 1);

    ny = ny{1};
    e = y{1}' - ny;

    e = e(:);

    for i = 1:numel(op)
        dfdx{i} = dfdx{i}(:);
    end

    dfdx = cell2mat(dfdx);
    np = op + (dfdx'*dfdx + 0.1*eye(numel(op)))\(dfdx'*e);

    if all(abs(np - op) < 1e-2)
        break;
    else
        op = np;
    end

end

mX = op;
cX = var(reshape(e, size(ny)));

end
