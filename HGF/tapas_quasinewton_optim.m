function optim = tapas_quasinewton_optim(f, init, varargin)
% This function implements the quasi-Newton minimization algorithm
% introduced by Broyden, Fletcher, Goldfarb, and Shanno (BFGS).
%
% INPUT:
%     f            Function handle of the function to be optimised
%     init         The point at which to initialize the algorithm
%     varargin     Optional settings structure that can contain the
%                  following fields:
%       tolGrad    Convergence tolerance in the gradient
%       tolArg     Convergence tolerance in the argument
%       maxIter    Maximum number of iterations
%       maxRegu    Maximum number of regularizations
%       verbose    Boolean flag to turn output on (true) or off (false)
%
% OUTPUT:
%     optim        Structure containing results in the following fields
%       valMin     The value of the function at its minimum
%       argMin     The argument of the function at its minimum
%       T          The inverse Hessian at the minimum calculated as a
%                  byproduct of optimization
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2019 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Dimension count
n = length(init);

% Defaults
verbose = false;
tolGrad = 1e-3;
tolArg  = 1e-3;
maxStep = 2;
maxIter = 1e3;
maxRegu = 4;
maxRst  = 4;
optIter = 0;

% Overrides
if nargin > 2
    options = varargin{1};
    
    if isfield(options,'tolGrad')
        tolGrad = options.tolGrad;
    end
    
    if isfield(options,'tolArg')
        tolArg = options.tolArg;
    end
    
    if isfield(options,'maxStep')
        maxStep = options.maxStep;
    end
    
    if isfield(options,'maxIter')
        maxIter = options.maxIter;
    end
    
    if isfield(options,'maxRegu')
        maxRegu = options.maxRegu;
    end
    
    if isfield(options,'maxRst')
        maxRst = options.maxRst;
    end

    if isfield(options,'verbose')
        verbose = options.verbose;
    end
    
    if isfield(options, 'optIter')
        optIter = options.optIter;
    end
end

% Make sure init is a column vector
if ~iscolumn(init)
    init = init';
    if ~iscolumn(init)
        error('tapas:hgf:QuasinewtonOptim:InitPointNotRow', 'Initial point has to be a row vector.');
    end
end

% preallocation (iterations of opt algo)
if optIter
    iter.x = NaN(maxIter+1, length(init));
    iter.val = NaN(maxIter+1, 1);
    iter.invH = struct;
    iter.rst = [];
else
    iter = [];
end

% Evaluate initial value of objective function
x = init;
val = f(x);
if optIter
    iter.x(1,:) = x';
    iter.val(1) = val;
end

if verbose
    disp(' ')
    disp(['Initial argument: ', num2str(x')])
    disp(['Initial value: ' num2str(val)])
end

% Calculate gradient
gradoptions.min_steps = 10;
grad = tapas_riddersgradient(f, x, gradoptions);

% Initialize negative Sigma (here called T) as the unit matrix
T = eye(n);
if optIter
    iter.invH(1).T = T;
end

% Initialize descent vector and slope
descvec = -grad';
slope   =  grad*descvec;

% Initialize new point and new value
newx    = NaN;
newval  = NaN;
dval    = NaN;

% Initialize reset count
resetcount = 0;

% Iterate
for i = 1:maxIter
    
    % Limit step size
    stepSize = sqrt(descvec'*descvec);
    if stepSize > maxStep
        descvec = descvec*maxStep/sqrt(descvec'*descvec);
    end
        
    regucount = 0;
    % Move in the descent direction, looping through regularizations
    for j = 0:maxRegu
        regucount = j;
        
        % Begin with t=1 and halve on each step
        t       = 0.5^j;
        newx    = x+t.*descvec;
        newval  = f(newx);

        % Regularize if the objective function value is Inf
        if isinf(newval)
            continue
        else
            dval = newval-val;
        end
        
        % Stop if the new value is sufficiently smaller
        if dval < 1e-4*t*slope
            break
        end
    end

    % Update point and value if regularizations have not been exhausted;
    % otherwise, reset and start again by jumping back 10% of the way to
    % the first initialization.
    if regucount < maxRegu
        dx   = newx-x;
        x    = newx;
        val  = newval;
        % Update step of optim algo
        if optIter
            iter.x(i+1,:) = x';
            iter.val(i+1) = val;
        end
    elseif resetcount < maxRst
        T       = eye(n);
        x       = x+0.1*(init-x);
        val     = f(x);
        % Update step of optim algo
        if optIter
            iter.x(i+1,:) = x';
            iter.val(i+1) = val;
            iter.invH(i+1).T = T;
            iter.rst = [iter.rst, i];
        end

        grad = tapas_riddersgradient(f, x, gradoptions);
        descvec = -grad';
        slope   =  grad*descvec;

        i = 0;
        resetcount = resetcount+1;

        if  verbose
            disp(' ')
            disp('Regularizations exhausted - resetting algorithm.')
            disp(['Initial argument: ', num2str(x')])
            disp(['Initial value: ' num2str(val)])
        end
        continue
    else
        disp(' ')
        disp('Warning: optimization terminated because the maximum number of resets was reached.')
        break
    end
    
    if verbose
        disp(' ')
        disp(['Argument: ', num2str(x')])
        disp(['Value: ' num2str(val)])
        disp(['Improvement: ' num2str(-dval)])
        disp(['Regularizations: ' num2str(regucount)])
    end

    % Test for convergence
    if max(abs(dx)./abs(max(x,1))) < tolArg
        if verbose
            disp(' ')
            disp('Converged on step size')
        end
        if optIter
            iter.x(i,:) = x';
            iter.val(i) = val;
            iter.invH(i).T = T;
        end
        break
    end
    
    % Update gradient
    oldgrad = grad;
    grad    = tapas_riddersgradient(f, x, gradoptions);
    dgrad   = grad-oldgrad;
    
    % Test for convergence
    if max(abs(grad').*max(abs(x),1)./max(abs(val),1)) < tolGrad
        if verbose
            disp(' ')
            disp('Converged on gradient size')
        end
        if optIter
            iter.x(i,:) = x';
            iter.val(i) = val;
            iter.invH(i).T = T;
        end
        break
    end

    % Update T according to BFGS
    if dgrad*dx > sqrt(eps*(dgrad*dgrad')*(dx'*dx))

        dgdx  = dgrad*dx;
        dgT   = dgrad*T;
        dgTdg = dgrad*T*dgrad';
        u     = dx/dgdx-dgT'/dgTdg;
        
        T = T + dx*dx'/dgdx - dgT'*dgT/dgTdg + dgTdg*(u*u');
    end
    
    % Update descent vector
    descvec = -T*grad';
    
    % Update slope
    slope = grad*descvec;
    
    % Update step of optim algo
    if optIter
        iter.invH(i+1).T = T;
    end
    
    % Warn if termination is only due to maximum of iterations being reached
    if i == maxIter
        disp(' ')
        disp('Warning: optimization terminated because the maximum number of iterations was reached.')
    end
end

% Collect results
optim.valMin = val;
optim.argMin = x;
optim.T      = T;
optim.iter   = iter;

return;
