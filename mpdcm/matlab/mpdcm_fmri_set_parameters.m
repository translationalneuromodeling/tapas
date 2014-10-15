function [theta] = mpdcm_fmri_set_parameters(p, theta, ptheta)
%% Sets the parameters introduced in vectorial form
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%


nt = numel(theta);
p = cell(size(theta));

for i = 1:nt
   
    tp = p{i};

    oi = 1;
    ni = sum(true(ptheta.A));

    theta.A(ptheta.A) = tp(oi:ni);
    
    oi = ni
    ni = oi + sum(true(ptheta.B));

    theta.B(ptheta.B) = tp(oi:ni);

    oi = ni
    ni = oi + sum(true(ptheta.C));

    theta.C(ptheta.C) = tp(oi:ni);


end

end
