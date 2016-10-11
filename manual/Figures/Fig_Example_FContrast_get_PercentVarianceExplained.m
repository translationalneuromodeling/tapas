function varianceExtraToAfterPercent = Fig_Example_FContrast_get_PercentVarianceExplained()
%computes variance explained in example single subject analysis on website,
% courtesy data of S. Iglesias
%
%   output = Fig_Example_FContrast_get_PercentVarianceExplained(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   Fig_Example_FContrast_get_PercentVarianceExplained
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-02-26
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
% $Id$
nScans = 854;
nColsDesignMatrix = 96;
nSess = 2;

nColsContrast = [2 2 4].*[3 4 1]*nSess;
F=[400 70 30];
nameContrast = {'Cardiac', 'Resp', 'CardXResp'};

for n = 1:3
    
    [varianceExtraToAfterPercent(n), varianceExplainedOfBeforePercent(n)] = ...
        convertFtoVarExplained(F(n), nScans, nColsDesignMatrix, ...
        nColsContrast(n));
    
    fprintf('variance explained by contrast %s: %4.1 \%\n', ...
        nameContrast{n}, varianceExtraToAfterPercent(n));
    
end
