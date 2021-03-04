function [h, montageSize] = tapas_uniqc_labeled_montage(varargin)
%MONTAGE Display multiple image frames as rectangular montage.
%   MONTAGE(FILENAMES) displays a montage of the images specified in
%   FILENAMES. FILENAMES is an N-by-1 or 1-by-N cell array of file name
%   strings. If the files are not in the current directory or in a
%   directory on the MATLAB path, specify the full pathname. (See the
%   IMREAD command for more information.) If one or more of the image files
%   contains an indexed image, MONTAGE uses the colormap from the first
%   indexed image file.
%
%   By default, MONTAGE arranges the images so that they roughly form a
%   square, but you can specify other arrangements using the 'Size'
%   parameter (see below).  MONTAGE creates a single image object to
%   display the images.
%
%   MONTAGE(I) displays a montage of all the frames of a multiframe image
%   array I. I can be a sequence of binary, grayscale, or truecolor images.
%   A binary or grayscale image sequence must be an M-by-N-by-1-by-K array.
%   A truecolor image sequence must be an M-by-N-by-3-by-K array.
%
%   MONTAGE(X,MAP) displays all the frames of the indexed image array X,
%   using the colormap MAP for all frames. X is an M-by-N-by-1-by-K array.
%
%   MONTAGE(..., PARAM1, VALUE1, PARAM2, VALUE2, ...) returns a customized
%   display of an image montage, depending on the values of the optional
%   parameter/value pairs. See Parameters below. Parameter names can be
%   abbreviated, and case does not matter.
%
%   H = MONTAGE(...) returns the handle of the single image object which
%   contains all the frames displayed.
%
%   Parameters
%   ----------
%   'Size'         A 2-element vector, [NROWS NCOLS], specifying the number
%                  of rows and columns in the montage. Use NaNs to have
%                  MONTAGE calculate the size in a particular dimension in
%                  a way that includes all the images in the montage. For
%                  example, if 'Size' is [2 NaN], MONTAGE creates a montage
%                  with 2 rows and the number of columns necessary to
%                  include all of the images.  MONTAGE displays the images
%                  horizontally across columns.
%
%                  Default: MONTAGE calculates the rows and columns so the
%                  images in the montage roughly form a square.
%
%   'Indices'      A numeric array that specifies which frames MONTAGE
%                  includes in the montage. MONTAGE interprets the values
%                  as indices into array I or cell array FILENAMES.  For
%                  example, to create a montage of the first four frames in
%                  I, use this syntax:
%
%                  montage(I,'Indices',1:4);
%
%                  Default: 1:K, where K is the total number of frames or
%                  image files.
%
%   'DisplayRange' A 1-by-2 vector, [LOW HIGH], that adjusts the display
%                  range of the images in the image array. The images
%                  must be grayscale images. The value LOW (and any value
%                  less than LOW) displays as black, the value HIGH (and
%                  any value greater than HIGH) displays as white. If you
%                  specify an empty matrix ([]), MONTAGE uses the minimum
%                  and maximum values of the images to be displayed in the
%                  montage as specified by 'Indices'. For example, if
%                  'Indices' is 1:K and the 'Display Range' is set to [],
%                  MONTAGE displays the minimum value in of the image array
%                  (min(I(:)) as black, and displays the maximum value
%                  (max(I(:)) as white.
%
%                  Default: Range of the datatype of the image array.
%   'LabelsIndices' Labels for each montage frame plotted
%                   default: Frame index
%   'FontSize',     Font size for labels (default: 10)
%
%   Class Support
%   -------------
%   A grayscale image array can be uint8, logical, uint16, int16, single,
%   or double. An indexed image array can be logical, uint8, uint16,
%   single, or double. MAP must be double. A truecolor image array can
%   be uint8, uint16, single, or double. The output is a handle to the
%   image object produced by MONTAGE.
%
%   Example 1
%   ---------
%   This example creates a montage from a series of images in ten files.
%   The montage has two rows and five columns.  Use the DisplayRange
%   parameter to highlight structures in the image.
%
%       fileFolder = fullfile(matlabroot,'toolbox','images','imdemos');
%       dirOutput = dir(fullfile(fileFolder,'AT3_1m4_*.tif'));
%       fileNames = {dirOutput.name}'
%       montage(fileNames, 'Size', [2 5]);
%
%       figure, montage(fileNames, 'Size', [2 5], ...
%       'DisplayRange', [75 200]);
%
%   Example 2
%   ---------
%   This example shows you how to customize the number of images in the
%   montage.
%
%       % Create a default montage.
%       load mri
%       montage(D, map)
%
%       % Create a new montage containing only the first 9 images.
%       figure
%       montage(D, map, 'Indices', 1:9);
%
%   See also IMMOVIE, IMSHOW, IMPLAY.

%   Copyright 1993-2011 The MathWorks, Inc.
%   $Revision$  $Date$

warning off images:imshow:magnificationMustBeFitForDockedFigure
warning off MATLAB:Figure:SetPosition

[I,cmap,mSize,indices,displayRange, labelsIndices, fontSize] = parse_inputs(varargin{:});

if isempty(indices) || isempty(I)
    hh = imshow([]);
    if nargout > 0
        h = hh;
    end
    return;
end

% Function Scope
nFrames = numel(indices);
nRows = size(I,1);
nCols = size(I,2);

montageSize = calculateMontageSize(mSize);

bigImage = createMontageImage;

if isempty(cmap)
    if isempty(displayRange)
        num = numel(I(:,:,:,indices));
        displayRange(1) = min(reshape(I(:,:,:,indices),[1 num]));
        displayRange(2) = max(reshape(I(:,:,:,indices),[1 num]));
    end
    hh = imshow(bigImage, displayRange, 'InitialMagnification', 'fit');
else
    hh = imshow(bigImage,cmap, 'InitialMagnification', 'fit');
end


addFrameLabels(montageSize, labelsIndices, nRows, nCols, nFrames, fontSize);

if nargout > 0
    h = hh;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function montageSize = calculateMontageSize(mSize)
        if nFrames == 1
            montageSize = [1 1];
        else
            if isempty(mSize) || all(isnan(mSize))
                %Calculate montageSize for the user
                
                % Estimate nMontageColumns and nMontageRows given the desired
                % ratio of Columns to Rows to be one (square montage).
                aspectRatio = 1;
                montageCols = sqrt(aspectRatio * nRows * nFrames / nCols);
                
                % Make sure montage rows and columns are integers. The order in
                % the adjustment matters because the montage image is created
                % horizontally across columns.
                montageCols = ceil(montageCols);
                montageRows = ceil(nFrames / montageCols);
                montageSize = [montageRows montageCols];
                
            elseif any(isnan(mSize))
                montageSize = mSize;
                nanIdx = isnan(mSize);
                montageSize(nanIdx) = ceil(nFrames / mSize(~nanIdx));
                
            elseif prod(mSize) < nFrames
                error(message('images:montage:sizeTooSmall'));
                
            else
                montageSize = mSize;
            end
            
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function b = createMontageImage
        
        nMontageRows = montageSize(1);
        nMontageCols = montageSize(2);
        nBands = size(I, 3);
        
        sizeOfBigImage = [nMontageRows*nRows nMontageCols*nCols nBands 1];
        if islogical(I)
            b = false(sizeOfBigImage);
        else
            b = zeros(sizeOfBigImage, class(I));
        end
        
        rows = 1 : nRows;
        cols = 1 : nCols;
        k = 1;
        
        for i = 0 : nMontageRows-1
            for j = 0 : nMontageCols-1,
                if k <= nFrames
                    b(rows + i * nRows, cols + j * nCols, :) = ...
                        I(:,:,:,indices(k));
                else
                    return;
                end
                k = k + 1;
            end
        end
        
    end

end %MONTAGE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I,cmap,montageSize,idxs,displayRange, labelsIndices, fontSize] = ...
    parse_inputs(varargin)

narginchk(1, 10);

% initialize variables
cmap = [];
montageSize = [];

charStart = find(cellfun('isclass', varargin, 'char'));

if iscell(varargin{1})
    %MONTAGE(FILENAMES.,..)
    [I,cmap] = getImagesFromFiles(varargin{1});
    
else
    %MONTAGE(I,...) or MONTAGE(X,MAP,...)
    I = varargin{1};
    validateattributes(varargin{1}, ...
        {'uint8' 'double' 'uint16' 'logical' 'single' 'int16'}, {}, ...
        mfilename, 'I, BW, or RGB', 1);
end

nframes = size(I,4);
displayRange = getrangefromclass(I);
idxs = 1:nframes;

fontSize = 10; % default font size
labelsIndices = cellfun(@(x) num2str(x), num2cell(idxs), 'UniformOutput', false);

if isempty(charStart)
    %MONTAGE(FILENAMES), MONTAGE(I) or MONTAGE(X,MAP)
    if nargin == 2
        %MONTAGE(X,MAP)
        cmap = validateColormapSyntax(I,varargin{2});
    elseif (nargin > 2)
        error(message('images:montage:nonCharParam'))
    end
    return;
end

charStart = charStart(1);

if charStart == 3
    %MONTAGE(X,MAP,Param1,Value1,...)
    cmap = validateColormapSyntax(I,varargin{2});
end

paramStrings = {'Size', 'Indices', 'DisplayRange', 'LabelsIndices', 'FontSize'};

for k = charStart:2:nargin
    
    param = lower(varargin{k});
    inputStr = validatestring(param, paramStrings, mfilename, 'PARAM', k);
    valueIdx = k + 1;
    if valueIdx > nargin
        error(message('images:montage:missingParameterValue', inputStr));
    end
    
    switch (inputStr)
        case 'FontSize'
            fontSize = varargin{valueIdx};
        case 'LabelsIndices'
            labelsIndices = varargin{valueIdx};
        case 'Size'
            montageSize = varargin{valueIdx};
            validateattributes(montageSize,{'numeric'},...
                {'vector','positive','numel',2}, ...
                mfilename, 'Size', valueIdx);
            
            montageSize = double(montageSize);
            
        case 'Indices'
            idxs = varargin{valueIdx};
            validateattributes(idxs, {'numeric'},...
                {'integer','nonnan'}, ...
                mfilename, 'Indices', valueIdx);
            
            if ~isempty(idxs)
                validateattributes(idxs, {'numeric'},{'vector'}, ...
                    mfilename, 'Indices', valueIdx);
            end
            
            invalidIdxs = ~isempty(idxs) && ...
                any(idxs < 1) || ...
                any(idxs > nframes);
            
            if invalidIdxs
                error(message('images:montage:invalidIndices'));
            end
            
            idxs = double(idxs);
            
        case 'DisplayRange'
            displayRange = varargin{valueIdx};
            displayRange = checkDisplayRange(displayRange, mfilename);
            
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cmap = validateColormapSyntax(I,cmap)

if isa(I,'int16')
    error(message('images:montage:invalidIndexedImage', 'double, single, or logical.'));
end

validateattributes(cmap,{'double'},{},mfilename,'MAP',1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I, map] = getImagesFromFiles(fileNames)

if isempty(fileNames)
    error(message('images:montage:invalidType'))
end

nframes = length(fileNames);

[img, map] = getImageFromFile(fileNames{1});
classImg = class(img);
sizeImg = size(img);

if length(sizeImg) > 2 && sizeImg(3) == 3
    nbands = 3;
else
    nbands = 1;
end

sizeImageArray = [sizeImg(1) sizeImg(2) nbands nframes];
if islogical(img)
    I = false(sizeImageArray);
else
    I = zeros(sizeImageArray, classImg);
end

I(:,:,:,1) = img;

for k = 2 : nframes
    [img,tempmap] = getImageFromFile(fileNames{k});
    
    if ~isequal(size(img),sizeImg)
        error(message('images:montage:imagesNotSameSize'));
    end
    
    if ~strcmp(class(img),classImg)
        error(message('images:montage:imagesNotSameClass'));
    end
    
    if isempty(map) && ~isempty(tempmap)
        map = tempmap;
    end
    I(:,:,:,k) = img;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function addFrameLabels(montageSize, labelsIndices, nRows, nCols, nFrames, ...
    fontSize)

if ~isempty(labelsIndices)
    nMontageRows = montageSize(1);
    nMontageCols = montageSize(2);
    
    hold all;
    k = 1;
    for i = 0 : nMontageRows-1
        for j = 0 : nMontageCols-1,
            if k <= nFrames
                text((j + 0.1) * nCols, (i + 0.05) * nRows,  ...
                    labelsIndices{k}, 'Color', [1 1 1], 'FontSize', fontSize);
            else
                return
            end
            k = k + 1;
        end
    end
end
end


function display_range = checkDisplayRange(display_range,fcnName)
%checkDisplayRange display range check function

%   Copyright 2006-2010 The MathWorks, Inc.
%   $Revision$  $Date$

if isempty(display_range)
    return
end

validateattributes(display_range, {'numeric'},...
    {'real' 'nonsparse' 'vector','nonnan'}, ...
    fcnName, '[LOW HIGH]', 2);

if numel(display_range) ~= 2
    error(message('images:checkDisplayRange:not2ElementVector'))
end

if display_range(2) <= display_range(1)
    error(message('images:checkDisplayRange:badDisplayRangeValues'))
end

display_range = double(display_range);

warning on images:imshow:magnificationMustBeFitForDockedFigure
warning on MATLAB:Figure:SetPosition
end


