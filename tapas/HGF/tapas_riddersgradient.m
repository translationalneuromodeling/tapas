function [gradf, err] = tapas_riddersgradient(f, x, varargin)
% Calculates the gradient of the function f at point x according to Ridders' method:
%
% Ridders, CJF. (1982). Accurate computation of F'(x) and F'(x) F''(x). Advances in Engineering
%     Software, 4(2), 75-6.
%
% INPUT:
%    f             Function handle of a real function of n real variables which are passed as
%                  *one* vector with n elements
%    x             Point at which to differentiate f
%
% OUTPUT:
%    gradf         Gradient of f at x (row vector)
%    err           Error estimates (row vector)
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

    n = length(x);

    % Defaults
    options.init_h     = 1;
    options.div        = 1.2;
    options.min_steps  = 3;
    options.max_steps  = 100;
    options.tf         = 2;
    gradf              = NaN(1,n);
    err                = NaN(1,n);
    
    % Overrides
    if nargin > 2
        options_ovr = varargin{1};

        if isfield(options_ovr,'init_h')
            options.init_h = options_ovr.init_h;
        end
        
        if isfield(options_ovr,'div')
            options.div = options_ovr.div;
        end
        
        if isfield(options_ovr,'min_steps')
            options.min_steps = options_ovr.min_steps;
        end
        
        if isfield(options_ovr,'max_steps')
            options.max_steps = options_ovr.max_steps;
        end
        
        if isfield(options_ovr,'tf')
            options.tf = options_ovr.tf;
        end
    end
    
    % Check if f and x match
    try
        f(x);
    catch err
        error('Function cannot be evaluated at differentiation point');
    end
    
    % Loop through argument variables
    for i = 1:n
        
        % Construct filehandle to be passed to riddersdiff
        fxi = @(xi) fxi(f,x,i,xi);
        
        % Calculate derivative
        [gradf(i), err(i)] = tapas_riddersdiff(fxi,x(i),options);
    end
end

function fxi = fxi(f,x,i,xi)
    xx    = x;
    xx(i) = xi;
    fxi   = f(xx);
end
