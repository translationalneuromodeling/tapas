function test_mpdcm_fmri_mle()
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%


d = test_mpdcm_fmri_load_td();


[y, u, theta, ptheta] = mpdcm_fmri_tinput(d{1});

mpdcm_fmri_mle(y, u, theta, ptheta);


end
