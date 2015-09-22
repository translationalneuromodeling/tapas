% Inverse cumulative density function of the logit-normal distribution.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: logitninv.m 14789 2012-03-05 09:06:58Z bkay $
% -------------------------------------------------------------------------
function x = logitninv(p,mu,sigma)
    assert((isscalar(mu) && isscalar(sigma)) || (isvector(mu) && isvector(sigma) && all(size(mu)==size(sigma))));
    assert(all(sigma>0),'sigma must be positive');
    assert(isscalar(p),'p must be a scalar');
    
    if p<0 | p>1
        x=NaN(1,length(mu));
    elseif p==0
        x=zeros(1,length(mu));
    elseif p==1
        x=ones(1,length(mu));
    else
        for i=1:length(mu)
            % Strategy: find root of logitncdf(x,mu,sigma) - p
            f = @(z) logitncdf(z,mu(i),sigma(i)) - p;
            x(i) = fzero(f,0.5);
        end
    end
    
    if size(mu,2)==1
        x = x';
    end
end
