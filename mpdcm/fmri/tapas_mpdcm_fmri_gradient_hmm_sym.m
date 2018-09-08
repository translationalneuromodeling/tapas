function [dfdx, ny] = tapas_mpdcm_fmri_gradient_hmm_sym(p, u, theta, ptheta, ...
    dx, dt, sloppy)
% Computes the numerical gradient using forward finate differences method.
% p is the location of the evalution of the gradient.
%
% Input:
% 
% p         -- Cell array of parameters in matrix form
% u         -- Cell array of experimental inputs.
% theta     -- Cell array of parameters in structure from
% ptheta    -- Structure of hyperparameters
% sloppy    -- Flag for not checking theta before integration. Defaults to
%           False.
% 
% Output:
%
% dfdx      -- Cell array of matrices with the gradients
% ny        -- Cell array of the matrices of evaluation a system in p
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

n = 5;

if nargin < n
    dx = 1;
end

n = n + 1;
if nargin < n
    dt = 1e-3;
end

n = n + 1;
if nargin < n
    sloppy = 1;
end

nx = theta{1}.dim_x;

sp1 = size(p, 2);

% Don't make a gradient for the noise
sp2 = numel(p{1}) - nx;

np = repmat(p, (2 * sp2 + 1), 1);

for j = 1:sp1
    for i = 2: sp2 + 1
        np{i, j}(i - 1) = np{i, j}(i - 1)  + dt;
    end
    for i = sp2 + 2: 2 * sp2 + 1
        np{i, j}(i - 1 - sp2) = np{i, j}(i - 1 - sp2)  - dt;
    end
end

np = reshape(np, 1, sp1 * (2 * sp2 + 1));

ntheta = repmat(theta, 2 * sp2 + 1, 1);
ntheta = reshape(ntheta, 1, sp1 * (2 * sp2 + 1));

ntheta = tapas_mpdcm_fmri_set_parameters(np, ntheta, ptheta);

ny = tapas_mpdcm_fmri_int(u, ntheta, ptheta, sloppy);
ny = reshape(ny, 2 * sp2 + 1, sp1);

dfdx = cell(1, sp1);

for j = 1:sp1
    dfdx(j) = {zeros(size(ny{j}, 1), size(ny{j}, 2), sp2)};
    for i = 1:sp2
        dfdx{j}(:, :, i) = 0.5 * (ny{i + 1, j} - ny{sp2 + i + 1, j})*(dx/dt);
    end
end

ny = ny(1, :);

end
