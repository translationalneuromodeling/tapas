function [ny] = tapas_mpdcm_fmri_predict(y, u, theta, ptheta, sloppy)
%% Makes prediction about the signal
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

n = 5;

if nargin < n
    sloppy = 0;
end

ny = tapas_mpdcm_fmri_int(u, theta, ptheta, sloppy); 

end

