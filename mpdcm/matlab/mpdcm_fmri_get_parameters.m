function [p] = mpdm_fmri_get_parameters(theta, ptheta)
%% Gets the parameters in vectorial form. 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%


nt = numel(theta);
p = cell(size(theta));

for i = 1:nt

    ta = theta.A(ptheta.A);  
    tb = theta.B(ptheta.B);
    tc = theta.C(ptheta.C);

    p{i} = [ta(:) tb(:) tc(:)];

end


end
