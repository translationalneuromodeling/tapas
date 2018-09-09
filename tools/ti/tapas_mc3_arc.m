function [v] = tapas_mc3_arc(ollh, olpp, T, mc3it)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

v = 1:numel(ollh);
nc = size(ollh, 2);

for l = 1:mc3it
    s = ceil(rand()*(nc - 1));
    p = min(1, exp(ollh(s) * T(s+1) + ollh(s+1) * T(s) ...
        - ollh(s) * T(s) - ollh(s+1) * T(s+1)));
    if rand() < p
        ollh([s, s+1]) = ollh([s+1, s]);
        olpp([s, s+1]) = olpp([s+1, s]);
        v([s, s+1]) = v([s+1, s]);
    end
end

end

