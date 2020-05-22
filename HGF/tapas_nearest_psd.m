function X = tapas_nearest_psd(X)
% Finds the nearest positive semi-defnite matrix to X
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% Input: X - a square matrix
%
% Output: X - nearest positive semi-definite matrix to input X

% Ensure symmetry
X = (X' + X)./2;

% Continue until X is positive semi-definite
while any(eig(X) < 0)
    % V: right eigenvectors, D: diagonalized X (X*V = V*D <=> X = V*D*V')
    [V, D] = eig(X);
    % Replace negative eigenvalues with 0 in D
    D = max(0, D);
    % Transform back
    X = V*D*V';
    % Ensure symmetry
    X = (X' + X)./2;
end

end

