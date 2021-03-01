classdef MrCopyData < handle
    % Provides a common clone-method for all object classes, plus
    % generalized find, compare and print-capabilities
    %
    % based on a posting by Volkmar Glauche
    % http://www.mathworks.com/matlabcentral/fileexchange/22965-clone-handle-object-using-matlab-oop
    %
    % heavily modified and extended by Lars Kasper and Saskia Klein
    
    % Author:   Saskia Klein & Lars Kasper
    % Created:  2010-04-15
    % Copyright (C) 2014 Institute for Biomedical Engineering
    %                    University of Zurich and ETH Zurich
    %
    % This file is part of the TAPAS UniQC Toolbox, which is released
    % under the terms of the GNU General Public Licence (GPL), version 3.
    % You can redistribute it and/or modify it under the terms of the GPL
    % (either version 3 or, at your option, any later version).
    % For further details, see the file COPYING or
    %  <http://www.gnu.org/licenses/>.
    %

    properties
    end
    methods
        function [this, unusedArgs] = MrCopyData(varargin)
            % Constructor for MrCopyData
            % either
            %
            %   this = MrCopyData for object with default values
            %
            %   OR
            %
            %   this = MrCopyData('empty') to create an object with all
            %   values set to [];
            %
            %   OR
            %   this = MrCopyData('param_name1', param_value1, 'param_name2', param_value2 )
            %          set of parameter names and values given, e.g.
            %          MrCopyData('dyn', 1)
            unusedArgs = {};
            if nargin
                if strcmpi(varargin{1}, 'empty')
                    this.clear();
                else
                    for cnt = 1:nargin/2 % save them to object properties
                        if isprop(this, varargin{2*cnt-1})
                        this.(varargin{2*cnt-1}) = varargin{2*cnt};
                        else % return unused props in a struct
                            unusedArgs = [unusedArgs, varargin(2*cnt-1:2*cnt)];
                        end
                    end
                end
            end
        end
    end
end