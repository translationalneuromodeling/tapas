classdef MrCopyDataTest < MrCopyData
    % Implements class with nested MrCopyData objects to test recursive
    % operations of MrCopyData
    %
    %
    % EXAMPLE
    %   MrCopyDataTest
    %
    %   See also MrCopyData
    
    % Author:   Lars Kasper & Saskia Bollmann
    % Created:  2017-07-20
    % Copyright (C) 2017 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public License (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    %

    
    properties
        numeric_cell
        string_cell
        numeric_vector
        string_value
        copydata_cell
        copydata_value
    end % properties
    
    
    methods
        
        function this = MrCopyDataTest(nRecursionDepth)
            % Constructor of class: this = MrCopyDataTest(nRecursionDepth)
            if nargin < 1
                nRecursionDepth = 2;
            end
            this.numeric_vector = rand(1,10);
            this.numeric_cell = num2cell(this.numeric_vector);
            this.string_value = num2str(rand());
            this.string_cell = cellfun(@(x) sprintf('string %d', x), ...
                this.numeric_cell, 'UniformOutput', false);
            if nRecursionDepth > 0
                % build a sub-object as property with reduced recusion
                % depth
                this.copydata_value = MrCopyDataTest(nRecursionDepth - 1);
                nElements = 5;
                this.copydata_cell = cell(1, nElements);
                for iElement = 1:nElements
                    this.copydata_cell{iElement} = MrCopyDataTest(nRecursionDepth - 1);
                end
            else
                this.copydata_value = [];
                this.copydata_cell = {};
            end
        end
        
        % NOTE: Most of the methods are saved in separate function.m-files in this folder;
        %       except: constructor, delete, set/get methods for properties.
        
    end % methods
    
end
