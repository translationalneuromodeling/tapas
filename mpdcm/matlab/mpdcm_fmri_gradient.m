function [dfdx ny] = mpdcm_fmri_gradient(p, u, theta, ptheta)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%


dt = 1e-5;

np = cell(1, numel(p)+1);
np(:) = {p};

for i = 2:numel(p)+1
    np{i}(i-1) = np{i}(i-1)  + dt;
end

ntheta = cell(1, numel(p)+1);
ntheta(:) = {theta{1}};

ntheta = mpdcm_fmri_set_parameters(np, ntheta, ptheta);

ny = mpdcm_fmri_int(u, ntheta, ptheta);

dfdx = cell(1, numel(p));

for i = 1:numel(p)
   dfdx{i} = (ny{i+1} - ny{1})/dt;
end

ny = ny(1);

end
