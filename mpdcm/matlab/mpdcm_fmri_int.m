function [y] = mpdcm_fmri_int(u, theta, ptheta, sloppy)
%% 
% sloppy -- Don't check the input
% tflag -- Test flag, this is done 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 4
    sloppy = 0;
end


if ~sloppy
    mpdcm_fmri_int_check_input(u, theta, ptheta);  
end

y = c_mpdcm_fmri_int(u, theta, ptheta);

end
