function [theta] = tapas_mpdcm_fmri_set_parameters(p, theta, ptheta)
% Sets the parameters introduced in vectorial form.
%
% Input:
% p     -- Cell array of parameters in matrix form
% theta     -- Cell array of parameters in structure form
% ptheta    -- Hyperparameters
%
% Ouput:
% thate     -- Cell array of parameters in structure form
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


nt = numel(theta);
nb = size(ptheta.X0, 2);
nl = numel(ptheta.Q);

for i = 1:nt

    nr = size(ptheta.a, 1);   
    tp = p{i};

    oi = 0;
    ni = ptheta.n_a;
    theta{i}.A(ptheta.i_a) = tp(oi + 1:ni);

    oi = ni;
    ni = oi + ptheta.n_b;
    theta{i}.B(ptheta.i_b) = tp(oi + 1: ni);
    theta{i}.vB = tp(oi + 1: ni);
        
    oi = ni;
    ni = oi + ptheta.n_c;
    theta{i}.C(ptheta.i_c) = tp(oi + 1:ni);

    oi = ni;
    ni = oi + ptheta.n_d;
    theta{i}.D(ptheta.i_d) = tp(oi + 1:ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.K = tp(oi + 1: ni);

    oi = ni;
    ni = oi + nr;
    theta{i}.tau(:) = tp(oi + 1: ni);

    oi = ni;
    ni = oi + 1;
    theta{i}.epsilon(:) = tp(oi + 1: ni);

    oi = ni;
    ni = oi + nl;
    theta{i}.lambda = tp(oi + 1: ni);

    oi = ni;
    ni = oi + nb * nr;
    theta{i}.beta = tp(oi + 1: ni);
    theta{i}.beta = reshape(theta{i}.beta, nb,  nr);

end

end


