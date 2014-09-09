function [hessf, err] = tapas_riddershessian(f, x, varargin)
% Calculates the hessian of the function f at point x according to Ridders' method:
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
%    hessf         Hessian of f at x
%    err           Error estimates
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
    hessf              = NaN(n);
    err                = NaN(n);
    
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
    
    % First: diagonal elements
    % Loop through argument variables
    for i = 1:n
        
        % Construct filehandle to be passed to riddersdiff2
        fxi = @(xi) fxi(f,x,i,xi);
        
        % Calculate derivative
        [hessf(i,i), err(i,i)] = tapas_riddersdiff2(fxi,x(i),options);
    end

    % Second: off-diagonal elements
    % Loop through argument variables
    for i = 2:n % rows
        for j = 1:i-1 % columns
        
            % Construct filehandle to be passed to riddersdiffcross
            fxixj = @(xixj) fxixj(f,x,i,j,xixj);
        
            % Calculate cross-derivative
            [hessf(i,j), err(i,j)] = tapas_riddersdiffcross(fxixj,[x(i),x(j)],options);
            hessf(j,i) = hessf(i,j);
            err(j,i)   = err(i,j);
        end
    end
end

function fxi = fxi(f,x,i,xi)
    xx    = x;
    xx(i) = xi;
    fxi   = f(xx);
end

function fxixj = fxixj(f,x,i,j,xixj)
    xx    = x;
    xx(i) = xixj(1);
    xx(j) = xixj(2);
    fxixj = f(xx);
end
