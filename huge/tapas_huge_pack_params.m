%% [ packedParameters ] = tapas_huge_pack_params( unpackedParameters, paramList )
% 
% Transform vectorized parameters into a cell array format.
%
% INPUT:
%       unpackedParameters - current value of parameters arranged as a
%                             vector
%       paramList          - supporting paramters
%
% OUTPUT:
%       packedParameters - current value of parameters arranged as a cell
%                          array 
% 

%
% Author: Sudhir Shankar Raman
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 
function [ packedParameters ] = tapas_huge_pack_params( unpackedParameters, paramList )

packedParameters{1} = reshape(unpackedParameters(...
    1:(paramList(3)*paramList(3))),paramList(3),paramList(3));
nextIndex = paramList(3)*paramList(3)+1;
packedParameters{2} = reshape(unpackedParameters(nextIndex:...
    (nextIndex+paramList(3)*paramList(4)-1)),paramList(3),paramList(4));
nextIndex = nextIndex+paramList(3)*paramList(4);
packedParameters{3} = reshape(unpackedParameters(nextIndex:...
    (nextIndex+paramList(4)*paramList(3)*paramList(3)-1)),paramList(3),...
    paramList(3),paramList(4));
nextIndex = nextIndex + paramList(3)*paramList(3)*paramList(4);
packedParameters{4} = reshape(unpackedParameters(nextIndex:...
    (nextIndex+paramList(3)*paramList(3)*paramList(3)-1)),paramList(3),...
    paramList(3),paramList(3));
nextIndex = nextIndex + paramList(3)*paramList(3)*paramList(3);
packedParameters{5} = reshape(unpackedParameters(nextIndex:...
    (nextIndex+paramList(3)*3-1)),paramList(3),3)';

return;