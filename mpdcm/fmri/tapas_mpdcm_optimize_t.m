function [nt] = tapas_mpdcm_optimize_t(ot, llh)
%% Optimizes the temperature schedule to more closely resemble the shape of 
% the data.
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

% If there is any odd value better not try anything
if any(isnan(llh(:))) || any(abs(llh(:)) == inf)
    nt = ot;
    return
end

nc = numel(ot);

% Let's try to find a workable schedule
for i = 1:nc - 4
    try
        [~, ~, nt] = tapas_genpath(ot, llh, i, 'pchip');
        % It worked
        nt = nt';
        break
    catch err

        % Ok keep going
    end
end

% It is a mess and we could not estimate anything
if i == nc - 4
    nt = ot;
else
    nt = [ot(1:i - 1) nt];
end


end

