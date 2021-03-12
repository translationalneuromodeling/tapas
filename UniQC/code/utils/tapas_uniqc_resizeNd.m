function x = tapas_uniqc_resizeNd(x,newsiz)

%RESIZE Resize any arrays and images.
%   Y = TAPAS_UNIQC_RESIZEND(X,NEWSIZE) resizes input array X using a DCT (discrete
%   cosine transform) method. X can be any array of any size. Output Y is
%   of size NEWSIZE.
%   
%   Input and output formats: Y has the same class as X.
%
%   Note:
%   ----
%   If you want to multiply the size of an RGB image by a factor N, use the
%   following syntax: TAPAS_UNIQC_RESIZEND(I,size(I).*[N N 1])
%
%   Examples:
%   --------
%     % Resize a signal
%     % original signal
%     x = linspace(0,10,300);
%     y = sin(x.^3/100).^2 + 0.05*randn(size(x));
%     % resized signal
%     yr = resize(y,[1 1000]);
%     plot(linspace(0,10,1000),yr)
%
%     % Upsample and downsample a B&W picture
%     % original image
%     I = imread('tire.tif');
%     % upsampled image
%     J = resize(I,size(I)*2);
%     % downsampled image
%     K = resize(I,size(I)/2);
%     % pictures
%     figure,imshow(I),figure,imshow(J),figure,imshow(K)
%
%     % Upsample and stretch a 3-D scalar array
%     load wind
%     spd = sqrt(u.^2 + v.^2 + w.^2); % wind speed
%     upsspd = resize(spd,[64 64 64]); % upsampled speed
%     slice(upsspd,32,32,32);
%     colormap(jet)
%     shading interp, daspect(size(upsspd)./size(spd))
%     view(30,40), axis(volumebounds(upsspd))
%
%     % Upsample and stretch an RGB image
%     I = imread('onion.png');
%     J = resize(I,size(I).*[2 2 1]);
%     K = resize(I,size(I).*[1/2 2 1]);
%     figure,imshow(I),figure,imshow(J),figure,imshow(K)
%
%   See also UPSAMPLE, RESAMPLE, IMRESIZE, DCTN, IDCTN
%
%   -- Damien Garcia -- 2009/11, revised 2010/11
%   website: <a
%   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>

error(nargchk(2,2,nargin));

siz = size(x);
N = prod(siz);

% avoid error of trailing ones in new size
nDims = numel(siz);
iDimNewSingleton = setdiff(find(newsiz==1),1:nDims);
newsiz(iDimNewSingleton) = [];

% Do nothing if size is unchanged
if isequal(siz,newsiz), return, end

% Do nothing if input is empty
if isempty(x), return, end

% Check size arguments
assert(isequal(length(siz),length(newsiz)),...
    'Number of dimensions must not change.')
newsiz = round(newsiz);
assert(all(newsiz>0),'Size arguments must be >0.')

class0 = class(x);
is01 = islogical(x);

% Deal with NaNs if any
I = isnan(x);
if any(I(:))
    % replace NaNs by nearest neighbors
    [~,L] = bwdist(~I);
    x(I) = x(L(I));
    % resize
    x = tapas_uniqc_resizeNd(x,newsiz);
    I = tapas_uniqc_resizeNd(I,newsiz);
    % reintroduce NaN
    x(I) = NaN;
    return
end

% DCT transform
x = dctn(x);

% Crop the DCT coefficients
for k = 1:ndims(x)
    siz(k) = min(newsiz(k),siz(k));
    x(siz(k)+1:end,:) = [];
    x = reshape(x,circshift(siz,[0 1-k]));
    x = shiftdim(x,1);
end

% Pad the DCT coefficients with zeros
x = padarray(x,max(newsiz-siz,zeros(size(siz))),0,'post');

% inverse DCT transform
x = idctn(x)*sqrt(prod(newsiz)/N);

% Back to the previous class
if is01, x = round(x); end
x = cast(x,class0);

end


%% DCTN
function y = dctn(y)

%DCTN N-D discrete cosine transform.
%   Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
%   the same size as X and contains the discrete cosine transform
%   coefficients. This transform can be inverted using IDCTN.
%
%   Reference
%   ---------
%   Narasimha M. et al, On the computation of the discrete cosine
%   transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
%
%   Example
%   -------
%       RGB = imread('autumn.tif');
%       I = rgb2gray(RGB);
%       J = dctn(I);
%       imshow(log(abs(J)),[]), colormap(jet), colorbar
%
%   The commands below set values less than magnitude 10 in the DCT matrix
%   to zero, then reconstruct the image using the inverse DCT.
%
%       J(abs(J)<10) = 0;
%       K = idctn(J);
%       figure, imshow(I)
%       figure, imshow(K,[0 255])
%
%   -- Damien Garcia -- 2008/06, revised 2011/11
%   -- www.BiomeCardio.com --

y = double(y);
sizy = size(y);
y = squeeze(y);
dimy = ndims(y);

% Some modifications are required if Y is a vector
if isvector(y)
    dimy = 1;
    if size(y,1)==1, y = y.'; end
end

% Weighting vectors
w = cell(1,dimy);
for dim = 1:dimy
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);
end

% --- DCT algorithm ---
if ~isreal(y)
    y = complex(dctn(real(y)),dctn(imag(y)));
else
    for dim = 1:dimy
        siz = size(y);
        n = siz(1);
        y = y([1:2:n 2*floor(n/2):-2:2],:);
        y = reshape(y,n,[]);
        y = y*sqrt(2*n);
        y = ifft(y,[],1);
        y = bsxfun(@times,y,w{dim});
        y = real(y);
        y(1,:) = y(1,:)/sqrt(2);
        y = reshape(y,siz);
        y = shiftdim(y,1);
    end
end
        
y = reshape(y,sizy);

end

%% IDCTN
function y = idctn(y)

%IDCTN N-D inverse discrete cosine transform.
%   X = IDCTN(Y) inverts the N-D DCT transform, returning the original
%   array if Y was obtained using Y = DCTN(X).
%
%   Reference
%   ---------
%   Narasimha M. et al, On the computation of the discrete cosine
%   transform, IEEE Trans Comm, 26, 6, 1978, pp 934-936.
%
%   Example
%   -------
%       RGB = imread('autumn.tif');
%       I = rgb2gray(RGB);
%       J = dctn(I);
%       imshow(log(abs(J)),[]), colormap(jet), colorbar
%
%   The commands below set values less than magnitude 10 in the DCT matrix
%   to zero, then reconstruct the image using the inverse DCT.
%
%       J(abs(J)<10) = 0;
%       K = idctn(J);
%       figure, imshow(I)
%       figure, imshow(K,[0 255])
%
%   See also DCTN, IDSTN, IDCT, IDCT2, IDCT3.
%
%   -- Damien Garcia -- 2009/04, revised 2011/11
%   -- www.BiomeCardio.com --

y = double(y);
sizy = size(y);
y = squeeze(y);
dimy = ndims(y);

% Some modifications are required if Y is a vector
if isvector(y)
    dimy = 1;
    if size(y,1)==1
        y = y.';
    end
end

% Weighing vectors
w = cell(1,dimy);
for dim = 1:dimy
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);
end

% --- IDCT algorithm ---
if ~isreal(y)
    y = complex(idctn(real(y)),idctn(imag(y)));
else
    for dim = 1:dimy
        siz = size(y);
        n = siz(1);
        y = reshape(y,n,[]);
        y = bsxfun(@times,y,w{dim});
        y(1,:) = y(1,:)/sqrt(2);
        y = ifft(y,[],1);
        y = real(y*sqrt(2*n));
        I = (1:n)*0.5+0.5;
        I(2:2:end) = n-I(1:2:end-1)+1;
        y = y(I,:);
        y = reshape(y,siz);
        y = shiftdim(y,1);            
    end
end
        
y = reshape(y,sizy);

end

