function [p] = mpdm_fmri_get_parameters(theta, ptheta)
%% Gets the parameters in vectorial form. 
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte
%
% Revision log:
%
%


nt = numel(theta);
p = cell(size(theta));

for i = 1:nt

    ta = theta{i}.A(logical(ptheta.a));  

    t = cell(numel(theta{i}.B), 1);
    for j = 1:numel(theta{i}.B);
        tt = theta{i}.B{j}(logical(ptheta.b(:,:,j)));
        t{j} = tt(:);
    end

    tb = cell2mat(cat(1, t(:)));
    tc = theta{i}.C(logical(ptheta.c));

    ttran = theta{i}.K;
    tdecay = theta{i}.tau;
    tepsilon = theta{i}.epsilon;
    tl = theta{i}.lambda;

    p{i} = [ta(:); tb(:); tc(:); ttran(:); tdecay(:); tepsilon(:); tl(:)];

end


end
