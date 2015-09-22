% Turns a Bayes factor into a prose representation of the strength of
% evidence. See: http://en.wikipedia.org/wiki/Bayes_factor
%
% Usage:
%     str = bf2str(BF)

% Kay H. Brodersen, ETH Zurich
% -------------------------------------------------------------------------
function str = bf2str(BF)
    
    % Check input
    assert(BF>0, 'BF cannot be negative');
    if (BF<1)
        BF = 1/BF;
        warning('Warning: transformed BF --> 1/BF');
    end
    
    % Turn into text
    if BF>=1, str = 'barely worth mentioning'; end
    if BF>=3, str = 'positive'; end
    if BF>=12, str = 'strong'; end
    if BF>=150, str = 'decisive'; end
    
end
