function [pks,locs] = tapas_physio_findpeaks(X,varargin)
%tapas_physio_findpeaks Find local peaks in data
%
% NOTE: This copy of function findpeaks is included in the TAPAS PhysIO 
%       Toolbox to make the dependency on Matlab's signal processing toolbox 
%       explicit. Please do not use this function if you haven't purchased
%       the signal processing toolbox.
%
%   PKS = tapas_physio_findpeaks(X) finds local peaks in the data vector X. A local peak
%   is defined as a data sample which is either larger than the two
%   neighboring samples or is equal to Inf.
%
%   [PKS,LOCS]= tapas_physio_findpeaks(X) also returns the indices LOCS at which the
%   peaks occur.
%
%   [...] = tapas_physio_findpeaks(X,'MINPEAKHEIGHT',MPH) finds only those peaks that
%   are greater than MINPEAKHEIGHT MPH. Specifying a minimum peak height
%   may help in reducing the processing time. MPH is a real valued scalar.
%   The default value of MPH is -Inf.
%
%   [...] = tapas_physio_findpeaks(X,'MINPEAKDISTANCE',MPD) finds peaks that are at
%   least separated by MINPEAKDISTANCE MPD. MPD is a positive integer
%   valued scalar. This parameter may be specified to ignore smaller peaks
%   that may occur in close proximity to a large local peak. For example,
%   if a large local peak occurs at index N, then all smaller peaks in the
%   range (N-MPD, N+MPD) are ignored. If not specified, MPD is assigned a
%   value of one. 
%
%   [...] = tapas_physio_findpeaks(X,'THRESHOLD',TH)finds peaks that are at least
%   greater than their neighbors by the THRESHOLD TH. TH is real valued
%   scalar greater than or equal to zero. The default value of TH is zero.
%
%   [...] = tapas_physio_findpeaks(X,'NPEAKS',NP) specifies the maximum number of peaks
%   to be found. NP is an integer greater than zero. If not specified, all
%   peaks are returned.
%
%   [...] = tapas_physio_findpeaks(X,'SORTSTR',STR) specifies the direction of sorting
%   of peaks. STR can take values of 'ascend','descend' or 'none'. If not
%   specified, STR takes the value of 'none' and the peaks are returned in
%   the order of their occurrence.
%
%   See also DSPDATA/FINDPEAKS

%   Copyright 2007-2010 The MathWorks, Inc.
%   $Revision: 235 $  $Date: 2013-08-19 18:28:07 +0200 (Mon, 19 Aug 2013) $

error(nargchk(1,11,nargin,'struct'));

[X,Ph,Pd,Th,Np,Str,infIdx] = parse_inputs(X,varargin{:});
[pks,locs] = getPeaksAboveMinPeakHeight(X,Ph);
[pks,locs] = removePeaksBelowThreshold(X,pks,locs,Th,infIdx);
[pks,locs] = removePeaksSeparatedByLessThanMinPeakDistance(pks,locs,Pd);
[pks,locs] = orderPeaks(pks,locs,Str);
[pks,locs] = keepAtMostNpPeaks(pks,locs,Np);

%--------------------------------------------------------------------------
function [X,Ph,Pd,Th,Np,Str,infIdx] = parse_inputs(X,varargin)

% Validate input signal
validateattributes(X,{'numeric'},{'nonempty','real','vector'},...
    'findpeaks','X');
M = numel(X);
if (M < 3)
    datamsgid = generatemsgid('emptyDataSet');
    error(datamsgid,'Data set must contain at least 3 samples.');
end

%#function dspopts.findpeaks
hopts = tapas_physio_uddpvparse('dspopts.findpeaks',varargin{:});
Ph  = hopts.MinPeakHeight;
Pd  = hopts.MinPeakDistance;
Th  = hopts.Threshold;
Np  = hopts.NPeaks;
Str = hopts.SortStr;

% Validate MinPeakDistance 
if ~isempty(Pd) && (~isnumeric(Pd) || ~isscalar(Pd) ||any(rem(Pd,1)) || (Pd < 1))
    Nmsgid = generatemsgid('invalidMinPeakDistance');
    error(Nmsgid,'MinPeakDistance should be an integer greater than 0.');
end

% Set default values for MinPeakDistance and NPeaks
if(isempty(Pd)), Pd = 1; end
if(isempty(Np)), Np = M; end

if(Pd >= M)
    pdmsgid = generatemsgid('largeMinPeakDistance');
    error(pdmsgid,'Invalid MinPeakDistance. Set MinPeakDistance as an integer in the range between 1 and %s.',...
        num2str(M));
end

% Replace Inf by realmax because the diff of two Infs is not a number
infIdx = isinf(X);
if any(infIdx),
    X(infIdx) = sign(X(infIdx))*realmax;
end
infIdx = infIdx & X>0; % Keep only track of +Inf

%--------------------------------------------------------------------------
function [pks,locs] = getPeaksAboveMinPeakHeight(X,Ph)

pks = [];
locs = [];

if all(isnan(X)),
    return,
end

Indx = find(X > Ph);
if(isempty(Indx))
    mphmsgid = generatemsgid('largeMinPeakHeight');
    warning(mphmsgid,'Invalid MinPeakHeight. There are no data points greater than MinPeakHeight.');
    return
end
    
% Peaks cannot be easily solved by comparing the sample values. Instead, we
% use first order difference information to identify the peak. A peak
% happens when the trend change from upward to downward, i.e., a peak is
% where the difference changed from a streak of positives and zeros to
% negative. This means that for flat peak we'll keep only the rising
% edge.
trend = sign(diff(X));
idx = find(trend==0); % Find flats
N = length(trend);
for i=length(idx):-1:1,
    % Back-propagate trend for flats
    if trend(min(idx(i)+1,N))>=0,
        trend(idx(i)) = 1; 
    else
        trend(idx(i)) = -1; % Flat peak
    end
end
        
idx  = find(diff(trend)==-2)+1;  % Get all the peaks
locs = intersect(Indx,idx);      % Keep peaks above MinPeakHeight
pks  = X(locs);

%--------------------------------------------------------------------------
function [pks,locs] = removePeaksBelowThreshold(X,pks,locs,Th,infIdx)

idelete = [];
for i = 1:length(pks),
    delta = min(pks(i)-X(locs(i)-1),pks(i)-X(locs(i)+1));
    if delta<Th,
        idelete = [idelete i]; %#ok<AGROW>
    end
end
if ~isempty(idelete),
    locs(idelete) = [];
end

X(infIdx) = Inf;                 % Restore +Inf
locs = union(locs,find(infIdx)); % Make sure we find peaks like [realmax Inf realmax]
pks  = X(locs);

%--------------------------------------------------------------------------
function [pks,locs] = removePeaksSeparatedByLessThanMinPeakDistance(pks,locs,Pd)
% Start with the larger peaks to make sure we don't accidentally keep a
% small peak and remove a large peak in its neighborhood. 

if isempty(pks) || Pd==1,
    return
end

% Order peaks from large to small
[pks, idx] = sort(pks,'descend');
locs = locs(idx);

idelete = ones(size(locs))<0;
for i = 1:length(locs),
    if ~idelete(i),
        % If the peak is not in the neighborhood of a larger peak, find
        % secondary peaks to eliminate.
        idelete = idelete | (locs>=locs(i)-Pd)&(locs<=locs(i)+Pd); 
        idelete(i) = 0; % Keep current peak
    end
end
pks(idelete) = [];
locs(idelete) = [];

%--------------------------------------------------------------------------
function [pks,locs] = orderPeaks(pks,locs,Str)

if isempty(pks), return; end

if strcmp(Str,'none')
    [locs idx] = sort(locs);
    pks = pks(idx);
else
    [pks,s]  = sort(pks,Str);
    locs = locs(s);
end

%--------------------------------------------------------------------------
function [pks,locs] = keepAtMostNpPeaks(pks,locs,Np)

if length(pks)>Np,
    locs = locs(1:Np);
    pks  = pks(1:Np);
end

% [EOF]
