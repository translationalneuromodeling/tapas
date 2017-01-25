function [lpp] = tapas_sem_seri_lpp_uniform(theta, ptheta)
%% Log prior probability of the parameters of the seri model. 
%
% Input
%   pv -- Parameters in numerical form.
%   theta -- Parameters of the model
%   ptheta -- Priors of parameters
%
% Output
%   lpp -- Log prior probability
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%


DIM_THETA = tapas_sem_seri_ndims();

lpp = zeros(1, numel(theta));

np = size(ptheta.jm, 2);
njm = tapas_zeromat(ptheta.jm);
njm = sum(njm, 2);

const = sum(njm .* -ptheta.uc);
 
for i = 1:numel(theta)
    lt = theta{i};

    if any(lt > ptheta.maxv) || any(lt < ptheta.minv)
        lpp(i) = -inf;
        continue
    end
    lpp(i) = const;
end

end %

function [out] = transform_input(in)

out = in;
it = [1:6 9:14 17:19];
out(it) = exp(it);
it = [8 9 16 16];
out(it) = atan(in(it))/pi + 0.5;

end
