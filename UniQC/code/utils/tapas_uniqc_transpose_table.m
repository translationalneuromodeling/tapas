function outputTable = tapas_uniqc_transpose_table(inputTable)
% transposes Matlab table object (rows->columns, columns->rows)
%
%   outputTable = tapas_uniqc_transpose_table(inputTable)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_transpose_table
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2020-09-23
% Copyright (C) 2020 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%

tempArray = table2array(inputTable);
outputTable = array2table(tempArray.');
outputTable.Properties.RowNames = inputTable.Properties.VariableNames;
outputTable.Properties.VariableNames = inputTable.Properties.RowNames;