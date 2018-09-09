%% [ DcmSpm ] = tapas_huge_export_spm( DcmInfo )
% 
% Convert DCM from DcmInfo format to SPM format.
% 
% INPUT:
%       DcmInfo - DCM in DcmInfo format
% 
% OUTPUT:
%       DcmSpm - cell array of DCM structs in SPM format
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
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
function [ DcmSpm ] = tapas_huge_export_spm( DcmInfo )
%% assemble DcmInfo
template = struct();
template.U = struct(); % place holder
template.Y = struct(); % place holder

N = DcmInfo.nSubjects; % #subjects
R = DcmInfo.nStates; % #regions
template.n = R; 

template.v = 0; % place holder

template.TE = DcmInfo.hemParam.echoTime;

template.options = struct();
template.options.nonlinear = DcmInfo.dcmTypeD;
template.options.two_state = false;
template.options.stochastic = false;
template.options.nograph = true;

% DCM connection indicators
template.a = DcmInfo.adjacencyA;
template.c = DcmInfo.adjacencyC;
template.b = DcmInfo.adjacencyB;
if(any(DcmInfo.dcmTypeD))    
    template.d = DcmInfo.adjacencyD;
else
    template.d = zeros(R,R,0);
end

% fill in data for individial subjects
DcmSpm = repmat({template},N,1);
for n = 1:N
    DcmSpm{n}.U.dt = DcmInfo.timeStep(n);
    DcmSpm{n}.U.u = DcmInfo.listU{n};
    DcmSpm{n}.Y.dt = DcmInfo.trSeconds(n);
    DcmSpm{n}.Y.y = reshape(DcmInfo.listBoldResponse{n}(:),[],R);
    DcmSpm{n}.Y.name = repmat({''},R,1);
    DcmSpm{n}.v = size(DcmSpm{n}.Y.y,1);
    DcmSpm{n}.Y.X0 = ones(DcmSpm{n}.v,1);
    DcmSpm{n}.Y.secs = DcmSpm{n}.v*DcmSpm{n}.Y.dt;
    DcmSpm{n}.Y.Q = cell(1,R);
    for r = 1:R
        base = zeros(R);
        base(r,r) = 1;
        tile = speye(DcmSpm{n}.v);
        DcmSpm{n}.Y.Q{r} = kron(base,tile);
    end
end




end
