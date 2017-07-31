function [llh] = tapas_sem_optimized_llh(y, u, theta, ptheta)
%% Computes the likelihood of the data using an optimized method.
%
% Input 
%
%   y       --  Observed behavioral data. A structure with fields 't' times
%               and 'a' action.
%   u       --  Experimental input. A structure with fields: 'tt' trial type, 
%               either prosaccade or antisaccade.
%   theta   --  Model parameters
%   ptheta  --  Priors
%
% Output
%
%   llh     --  Log likelilhood
%

% aponteeduardo@gmail.com
% copyright (C) 2015
%

% Start the data to a compatible format.
data.y = y;
data.u = u;

% Verify that theta is in the right order. 
[ns, nc] = size(theta);

if ns ~= 1
    error('tapas:sem', ...
        'First dimension of theta should be 1, instead %d', ns);
end

% Compute the loglikelihood at once.
llh = ptheta.method(data, theta);


end
