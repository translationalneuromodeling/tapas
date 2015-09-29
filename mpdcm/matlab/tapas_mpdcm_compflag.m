function [compflag] = tapas_mpdcm_compflag()
%% Returns the precision in which mpdcm has been compiled.
%
% Input
%
% Output
%   compflag        -- 0 if mpdcm has been compiled in single precision and 1
%                   if it has been compiled in double precision.
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%


compflag = c_mpdcm_compflag();


end
