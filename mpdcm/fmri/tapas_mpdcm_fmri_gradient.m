function [dfdx, ny] = tapas_mpdcm_fmri_gradient(p, u, theta, ptheta, sloppy)
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

if nargin < 5
    sloppy = 0;
end

dt = 1e-5;

sp1 = size(p, 1);
sp2 = numel(p{1});

np = repmat(p, 1, sp2+1);

for j = 1:sp1
    for i = 2:sp2+1
        np{j, i}(i-1) = np{j, i}(i-1)  + dt;
    end
end

ntheta = repmat(theta, 1, sp2 + 1);
ntheta = tapas_mpdcm_fmri_set_parameters(np, ntheta, ptheta);

ny = tapas_mpdcm_fmri_int(u, ntheta, ptheta, sloppy);

dfdx = cell(sp1, 1);
for i = 1:sp1
    dfdx(i) = {zeros(size(ny{i}, 1), size(ny{i}, 2), numel(p{1}))};
end
for j = 1:sp1
    for i = 1:sp2
        dfdx{j}(:, :, i) = (ny{j, i+1} - ny{j, 1})/dt;
    end
end

ny = ny(:, 1);

end
