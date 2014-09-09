function [df, err] = tapas_riddersdiff(f, x, varargin)
% Differentiates the function f at point x according to Ridders' method:
%
% Ridders, CJF. (1982). Accurate computation of F'(x) and F'(x) F''(x). Advances in Engineering
%     Software, 4(2), 75-6.
%
% INPUT:
%    f             Function handle of a scalar real function of one real variable
%    x             Point at which to differentiate f
%
% OUTPUT:
%    df            First derivative of f at x
%    err           Error estimate
%
% OPTIONS:
%    Optionally, the third argument of the function can be a structure containing further
%    settings for Ridder's method.
%
%    varargin{1}.init_h      Initial finite difference (default: 1)
%    varargin{1}.div         Divisor used to reduce h on each step (default: 1.2)
%    varargin{1}.min_steps   Minimum number of steps in h (default: 3)
%    varargin{1}.max_steps   Maximum number of steps in h (default: 100)
%    varargin{1}.tf          Terminate if last step worse than preceding by a factor of tf
%                            (default: 2)
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is released under the terms of the GNU General Public Licence (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either version 3 or, at your option,
% any later version). For further details, see the file COPYING or <http://www.gnu.org/licenses/>.
    
    % Defaults
    init_h     = 1;
    div        = 1.2;
    min_steps  = 3;
    max_steps  = 100;
    tf         = 2;
    df         = NaN;
    err        = realmax;
    
    % Overrides
    if nargin > 2
        options = varargin{1};

        if isfield(options,'init_h')
            init_h = options.init_h;
        end
        
        if isfield(options,'div')
            div = options.div;
        end
        
        if isfield(options,'min_steps')
            min_steps = options.min_steps;
        end
        
        if isfield(options,'max_steps')
            max_steps = options.max_steps;
        end
        
        if isfield(options,'tf')
            tf = options.tf;
        end
    end
    
    % Initialize matrix of polynomial interpolation values
    P = NaN(max_steps);
    
    % Initialize finite difference step
    h = init_h;
        
    % Approximate derivative at initial step
    P(1,1) = (f(x+h)-f(x-h))/(2*h);
    
    % Loop through rows of P (i.e., steps of h)
    for i = 2:max_steps
        
        % New step size
        h = h/div;
        
        % Approximate derivative at this step
        P(i,1) = (f(x+h)-f(x-h))/(2*h);
        
        % Use square of div for extrapolation because errors increase
        % quadratically with h (here, of course, they decrease quadratically
        % because we're reducing h...)
        divsq = div^2;
        t = divsq;
        
        % Fill the current row using Richardson extrapolation
        for j = 2:i
            
            % Richardson
            P(i,j) = (t*P(i,j-1)-P(i-1,j-1))/(t-1);
            
            % Increment extrapolation factor
            t = t*divsq;
            
            % Error on this trial is defined as the maximum absolute difference
            % to the extrapolation parents
            currerr = max(abs(P(i,j)-P(i,j-1)),abs(P(i,j)-P(i-1,j-1)));
            
            if currerr < err
                err = currerr;
                df = P(i,j);
            end
        end

        % Stop if errors start increasing (to be expected for very small
        % values of h)
        if i > min_steps && abs(P(i,i)-P(i-1,i-1)) > tf*err
            return
        end
    end
end
