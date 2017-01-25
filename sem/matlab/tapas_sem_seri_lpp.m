function [lpp] = tapas_sem_seri_lpp(theta, ptheta)
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

[lpp] = tapas_sem_prosa_lpp(theta, ptheta);

end
