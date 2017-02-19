function [njm] = tapas_zeromat(jm)
%% Zeros all but the first entry in each column of jm.
%
% Input
%   jm      -- Matrix.
%
% Output
%   njm     -- Matrix.
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%

njm = zeros(size(jm));

j = 0;
i = 0;

while j < size(jm, 2)
    j = j + 1;
    i = 0;
    while i < size(jm, 1)
        i = i + 1;
        if jm(i, j);
            njm(i, j) = 1;
            break
        end
    end
end
end

