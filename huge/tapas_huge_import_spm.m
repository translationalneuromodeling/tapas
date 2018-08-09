%% [ DcmInfo ] = tapas_huge_import_spm( DcmSpm )
% 
% Convert DCMs from SPM format to DcmInfo format.
% 
% INPUT:
%       DcmSpm - (cell) array of DCM structs in SPM format. All DCM in
%                DcmSpm must have same structure (i.e.: same number of
%                regions and same connections.) They may have different
%                number of scans.
% 
% OUTPUT:
%       DcmInfo - DCM in DcmInfo format
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
function [ DcmInfo ] = tapas_huge_import_spm( DcmSpm )
%% check input
if isvector(DcmSpm)&&isstruct(DcmSpm)
    % convert to cell array
    try
        DcmSpm = {DcmSpm(:).DCM}';
    catch
        DcmSpm = num2cell(DcmSpm);
    end
else
    assert(iscell(DcmSpm),'TAPAS:HUGE:InputFormat',...
        'Input must be cell array or array of structs');
end


%% assemble DcmInfo
DcmInfo = struct();

N = length(DcmSpm); % #subjects
DcmInfo.nSubjects = N;
R = DcmSpm{1}.n; % #regions
DcmInfo.nStates = R;
L = size(DcmSpm{1}.U.u,2); % #inputs
DcmInfo.nInputs = L;

% DCM connection indicators
DcmInfo.adjacencyA = logical(DcmSpm{1}.a);
DcmInfo.adjacencyC = logical(DcmSpm{1}.c);
DcmInfo.adjacencyB = logical(DcmSpm{1}.b);
if(~isempty(DcmSpm{1}.d))    
    DcmInfo.adjacencyD = DcmSpm{1}.d;
else
    DcmInfo.adjacencyD = false(R,R,R);
end
DcmInfo.dcmTypeB = any(DcmInfo.adjacencyB(:));
DcmInfo.dcmTypeD = any(DcmInfo.adjacencyD(:));

adjacencyList = [(DcmInfo.adjacencyA(:))' (DcmInfo.adjacencyC(:))'];
adjacencyList = [adjacencyList (DcmInfo.adjacencyB(:))']; 
adjacencyList = [adjacencyList (DcmInfo.adjacencyD(:))']; 
connectionIndicator = find(adjacencyList ~= 0);
noConnectionIndicator = find(adjacencyList == 0);
DcmInfo.noConnectionIndicator = noConnectionIndicator;
DcmInfo.connectionIndicator = connectionIndicator;
DcmInfo.nNoConnections = length(noConnectionIndicator);
DcmInfo.nConnections = length(connectionIndicator);

nParameters = (R^2)*(1 + L + R) + L*R + 3*R; % A, B, D, C, hem 
DcmInfo.nParameters = nParameters;



% hemodynamic model
DcmInfo.hemParam = struct();
% decay (kappa), transit time (tau), ratio intra/extra (epsilon)
DcmInfo.hemParam.listHem = [0.64,2,1]; % SPM default values
DcmInfo.hemParam.scaleC = DcmSpm{1}.Y.dt/DcmSpm{1}.U.dt;
if isfield(DcmSpm{1},'TE')
    DcmInfo.hemParam.echoTime = DcmSpm{1}.TE;
else
    DcmInfo.hemParam.echoTime = 0.04; % default value hard-coded into SPM
end
% default value hard-coded into SPM
DcmInfo.hemParam.restingVenousVolume = 4;
DcmInfo.hemParam.relaxationRateSlope = 25;
DcmInfo.hemParam.frequencyOffset = 40.3;
DcmInfo.hemParam.oxygenExtractionFraction = .4;
DcmInfo.hemParam.rho = 4.3;
DcmInfo.hemParam.gamma = .32;
DcmInfo.hemParam.alphainv = 1/.32;
DcmInfo.hemParam.oxygenExtractionFraction2 = .4;

% inputs and data
DcmInfo.trSeconds = zeros(1,N);
DcmInfo.trSteps = zeros(1,N);
DcmInfo.timeStep = zeros(1,N);
DcmInfo.nTime = zeros(1,N);
    
DcmInfo.listU = cell(1,N);
DcmInfo.listBoldResponse = cell(1,N);
DcmInfo.listResponseTimeIndices = cell(1,N);


DcmInfo.listParameters = cell(N,1);
DcmInfo.trueLabels = NaN(1,N);

for n = 1:N
    
    assert(DcmSpm{n}.n==R,'TAPAS:HUGE:InconsistentParameters',...
        'Number of regions inconsistent between subjects');
    assert(size(DcmSpm{n}.U.u,2)==L,'TAPAS:HUGE:InconsistentParameters',...
        'Number of inputs inconsistent between subjects');
    
    assert(~any(DcmSpm{n}.a(~DcmInfo.adjacencyA)),...
        'TAPAS:HUGE:InconsistentParameters',...
        'A matrix inconsistent between subjects');
    assert(~any(DcmSpm{n}.b(~DcmInfo.adjacencyB)),...
        'TAPAS:HUGE:InconsistentParameters',...
        'B matrix inconsistent between subjects');    
    assert(~any(DcmSpm{n}.c(~DcmInfo.adjacencyC)),...
        'TAPAS:HUGE:InconsistentParameters',...
        'C matrix inconsistent between subjects');    
    if(~isempty(DcmSpm{n}.d))    
    	adjacencyD = DcmSpm{n}.d;
    else
    	adjacencyD = false(R,R,R);
    end
    assert(~any(adjacencyD(~DcmInfo.adjacencyD)),...
        'TAPAS:HUGE:InconsistentParameters',...
        'D matrix inconsistent between subjects');

    % inputs
    DcmInfo.listU{n} = full(DcmSpm{n}.U.u);
    DcmInfo.nTime(n) = size(DcmInfo.listU{n},1);
    DcmInfo.timeStep(n) = DcmSpm{n}.U.dt;
    
    % BOLD time series
    [nScans,nRegions] = size(DcmSpm{n}.Y.y);
    assert(nRegions==R,'TAPAS:HUGE:DataSize',...
        'Size of data vector inconsistent with number of regions');
    DcmInfo.listBoldResponse{n} = DcmSpm{n}.Y.y(:);
    DcmInfo.trSeconds(n) = DcmSpm{n}.Y.dt;
    DcmInfo.trSteps(n) = DcmInfo.trSeconds(n)/DcmInfo.timeStep(n);
    DcmInfo.listResponseTimeIndices{n} = ...
        DcmInfo.trSteps(n):DcmInfo.trSteps(n):DcmInfo.trSteps(n)*nScans;
    
end


end
