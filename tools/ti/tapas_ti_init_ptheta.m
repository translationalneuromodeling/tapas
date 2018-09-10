function [nptheta] = init_ptheta(ptheta, htheta, pars)
% Precompute certain quantities

nptheta = ptheta;

if isfield(ptheta, 'prepare_ptheta')
    nptheta = ptheta.prepare_ptheta(ptheta, htheta, pars);
end

if ~isfield(ptheta, 'sm')
    nptheta.sm = tapas_ti_zeromat(nptheta.jm);
end


end 
