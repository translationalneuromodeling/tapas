function PPfun = ppcreate(varargin)
%PPCREATE Create 1-D Piecewise Polynomial Function.
% PPfun = PPCREATE(X,Y,Type,P,Tag) creates a function handle PPfun for a
% 1-D piecewise polynomial described by the data X and Y.
% Type is a character string identifying the type of piecewise
% polynomial desired.
% P is an optional array of parameters required for some Types.
% Tag is an optional character string to store with PPfun for
% identification purposes.
%
% Type        Description
% 'pchip'     same as MATLAB PCHIP function, no P required.
% 'spline'    same as MATLAB SPLINE function, no P required.
% 'notaknot'  same as MATLAB SPLINE function, no P required.
% 'extrap'    same as MATLAB SPLINE function, no P required.
% 'natural'   spline, y''=0 at end points, no P required.
% 'parabolic' spline, first and last polys are parabolic, no P required.
% 'clamped'   spline, P is two element vector of slopes y' at end points.
% 'curvature' spline, P is two element vector of y'' at the end points.
% 'hermite'   spline, P is a vector of slopes at all points in X.
%
% PPfun = PPCREATE(PP,Tag) where PP is the 1-D piecewise polynomial
% structure returned by PP = PCHIP(X,Y) or PP = SPLINE(X,Y) creates a
% function handle PPfun containing PP. Tag is an optional string as
% described above.
%
% PPfun2 = PPCREATE(PPfun,Op,P,Tag) peforms the operation specified by the
% string Op on the piecewise polynomial function handle PPfun, returning a
% new function handle PPfun2. P is an optional parameter. Tag is an
% optional string as described above.
%
% Operation    Description
% diff         differentiate the piecewise polynomial
% int          integrate the piecewise polynomial, P is the integration
%                 constant or initial value of the integral.
% tag          store tag string P with function handle.
% inv          inverse interpolation function handle returned, e.g., PPinv,
%                 interpolates the piecewise polynomial, such that
%                 [Xi,Yi]=PPinv(Yi) finds all points Xi where the piecewise
%                 polynomial has the scalar value Yi. If none are found an
%                 empty array is returned. On output Yi=repmat(Yi,size(Xi)).
%                 PPinv('tag') returns the tag assigned to PPinv.
% cut          cut spline apart, P = [Xmin Xmax] defines the range of the
%                 revised piecewise polynomial.
%
%--------------------------------------------------------------------------
% The created piecewise polynomial function handle PPfun has the following
% features:
% Sytax           Description
%
% PPfun(x)        evaluate the function at the points in x.
%
% PPfun('tag')    return Tag string stored in PPfun.
% PPfun('pp')     return MATLAB-defined piecewise polynomial structure.
% PPfun('breaks') return breakpoints X used to create PPfun.
% PPfun('pieces') return number of piecewise polynomials, length(X)-1.
% PPfun('order')  return order of the piecewise polynomial. (4 = cubic)
% PPfun('plot')   plots the piecewise polynomial over its entire range.
%
% See also: SPLINE, PCHIP, MKPP, UNMKPP, PPVAL.

% D.C. Hanselman, University of Maine, Orono, ME 04469
% MasteringMatlab@yahoo.com
% Mastering MATLAB 7
% 2006-03-23

% Modified by Eduardo Aponte 2017

if nargin<1
   error('At Least One Input is Required.')
end
PPData=struct('pp',[],'tag','');

switch class(varargin{1})
case 'struct'                                                % PPCREATE(PP)
   pp=varargin{1};
   if isstruct(pp) && isfield(pp,'form')...
                   && strcmp(pp.form,'pp') && (pp.dim==1)
      PPData.pp=varargin{1};
      if nargin==2 && ischar(varargin{2})
         PPData.tag=varargin{2};
      end
      PPfun=@(x) PPfunction(PPData,x);      
   else
      error('1-D PP Structure Expected.')
   end
   
case 'function_handle'                               % PPCREATE(PPfun,Op,P)
   PPfun=varargin{1};
   pp=PPfun('pp');
   if nargin<2
      error('At Least Two Input Arguments Required.')
   elseif ~ischar(varargin{2})
      error('Second Input Must be an Operation String.')
   end
   switch lower(varargin{2}(1:min(3,length(varargin{2}))))
      
   case 'tag'                                        % tag tag tag
      if nargin~=3 || ~ischar(varargin{3})
         error('Tag String Expected.')
      end
      PPData.tag=varargin{3};
      PPData.pp=pp;
      PPfun=@(x) PPfunction(PPData,x);
      
   case 'dif'                                        % differentiate
      if pp.order==0
         error('Piecewise Polynomial Cannot be Differentiated.')
      end
      pp.coefs=repmat(pp.order-1:-1:1,pp.pieces,1).*pp.coefs(:,1:end-1);
      pp.order=pp.order-1;
      PPData.pp=pp;
      if nargin==3 && ischar(varargin{3})
         PPData.tag=varargin{3};
      else
         PPData.tag='Derivative';
      end
      PPfun=@(x) PPfunction(PPData,x);
      
   case 'int'                                        % integrate
      if nargin==2 || ischar(varargin{3}) || isempty(varargin{3})
         c=0;
      elseif isnumeric(varargin{3}) 
         c=varargin{3}(1);
      else
         error('Numeric Integration Constant Expected.')
      end
      sf=repmat(pp.order:-1:1,pp.pieces,1);
      ico=[pp.coefs./sf zeros(pp.pieces,1)];
      pp.order=pp.order+1;
      dx=diff(pp.breaks(:));
      tmp=ico(:,1);
      for i=2:pp.order % make integral cumulative
         tmp=dx.*tmp + ico(:,i);
      end
      ico(:,end)=cumsum([c;tmp(1:end-1)]);
      pp.coefs=ico;
      PPData.pp=pp;
      if ischar(varargin{nargin})
         PPData.tag=varargin{nargin};
      else
         PPData.tag='Integral';
      end
      PPfun=@(x) PPfunction(PPData,x);
   case 'cut'                                      % cut cut cut
      if nargin<3 || ~isnumeric(varargin{3}) || numel(varargin{3})~=2
         error('P = [Xmin Xmax] Argument Expected.')
      end
      xmin=min(varargin{3});
      xmax=max(varargin{3});
      imin=find(xmin<pp.breaks,1);
      if isempty(imin)
         imin=1;
      else
         imin=max(imin-1,1);
      end
      imax=find(xmax<pp.breaks,1);
      if isempty(imax)
         imax=length(pp.breaks);
      end
      pp.breaks=pp.breaks(imin:imax);
      pp.coefs=pp.coefs(imin:imax-1,:);
      pp.pieces=length(pp.breaks)-1;
      if ischar(varargin{nargin})
         PPData.tag=varargin{nargin};
      else
         PPData.tag='Cut';
      end
      PPData.pp=pp;
      PPfun=@(x) PPfunction(PPData,x);

   case 'inv'                                    % inverse function handle
      PPData.pp=pp;
      if nargin==3 && ischar(varargin{3})
         PPData.tag=varargin{3};
      else
         PPData.tag='InverseInterpolation';
      end
      PPfun=@(y) PPInverse(PPData,y);

   otherwise
      error('Unknown Operation Selected.')
   end
   
case {'single' 'double'}                                % PPCREATE(X,Y,...)
   x=varargin{1};
   if nargin<2
      error('At Least Two Numeric Vectors X and Y Are Required.')
   end
   y=varargin{2};
   if ~isreal(x)
      error('X Must Contain Real Values.')
   end
   [x,idx]=sort(x(:)); % put x in increasing order
   nx=length(x);
   if nx~=numel(y);
      error('X and Y Must Contain the Same Number of Elements.')
   end
   y=reshape(y(idx),size(x));
   if nx<4
      error('At Least 4 Pairs of Data Points are Required.')
   end
   n=nx-1;        % number of pieces
   H=diff(x);     % differences in x
   if any(H==0)
      error('X Values Must be Distinct.')
   end
   D=diff(y)./H;  % slopes
   pptypes={'pchip';'spline';'notaknot';'extrap';...
      'natural';'parabolic';'clamped';'curvature';'hermite'};
   if nargin==2
      pptype='spline';
   elseif ischar(varargin{3}) && ~isempty(varargin{3}) && ...
         any(strncmpi(varargin{3},pptypes,2))
      pptype=pptypes{strncmpi(varargin{3},pptypes,2)};
   else
      error('Valid Character String Type Expected.')
   end
   if nargin>3 && ischar(varargin{end})
      PPData.tag=varargin{end};
   else % default tag is selected Type
      PPData.tag=pptype;
   end
   if any(strncmpi(pptype,{'c','h'},1)) % P required
      if nargin<4 || ischar(varargin{4})
         error('Parameter Vector P Required.')
      end
      P=varargin{4};
   end
   pp=struct('form','pp','breaks',x.',...     % create default PP structure
             'coefs',zeros(n,4),'pieces',n,'order',4,'dim',1);
   pptype=lower(pptype(1:2));
          
   if pptype(1)=='h'                          % 'hermite' requested
      if nargin<4 || ischar(varargin{4}) || ...
            numel(varargin{4})~=nx || ~isnumeric(varargin{4})
         error('Vector of Slopes the Same Size as X Required.')
      end
      dy=varargin{4}(:);
      pp.coefs(:,4)=y(1:end-1); % compute coefficients
      pp.coefs(:,3)=dy(1:end-1);
      pp.coefs(:,2)=(3*D - (2*dy(1:end-1)+dy(2:end)))./H;
      pp.coefs(:,1)=(-2*D + (dy(2:end)+dy(1:end-1)))./(H.*H);
      PPData.pp=pp;
      PPfun = @(x) PPfunction(PPData,x);
      
   elseif strcmp(pptype,'pc')                % MATLAB pchip
      PPData.pp = pchip(x,y);
      PPfun = @(x) PPfunction(PPData,x);    
   else                                      % spline construction
      
      V=[0;6*diff(D);0];              % right side vector
      B=[0;2*(H(1:n-1)+H(2:n));0];    % diagonal elements
      A=[0;H(2:n)];                   % subdiagonal elements
      C=A;                            % superdiagonal elements
      switch pptype
      case {'sp' 'no' 'ex'}  % MATLAB spline
         B(1)=H(2);
         B(2)=B(2)+H(1)+H(1)*H(1)/H(2);
         B(n)=B(n)+H(n)+H(n)*H(n)/H(n-1);
         B(nx)=H(n-1);
         C(1)=-H(1)-H(2);
         C(2)=C(2)-H(1)*H(1)/H(2);
         C(n)=0;
         A(n)=-H(n-1)-H(n);
         A(n-1)=A(n-1)-H(n)*H(n)/H(n-1);
         V(1)=0;
         V(nx)=0;   
      case 'na' % natural spline
         B(1)=1;
         B(nx)=1;
         C(n)=0;
         A(n)=0;
         V(1)=0;
         V(nx)=0;
      case 'pa' % parabolic spline
         B(1)=1;
         B(2)=B(2)+H(1);
         B(n)=B(n)+H(n);
         B(nx)=-1;
         C(1)=-1;
         C(n)=0;
         A(n)=1;
         V(1)=0;
         V(nx)=0;
      case 'cl' % clamped spline
         B(1)=1;
         B(2)=B(2)-H(1)/2;
         B(n)=B(n)-H(n)/2;
         B(nx)=1;
         C(1)=0.5;
         C(n)=0;
         A(n)=0.5;
         V(1)=3*(D(1)-P(1))/H(1);
         V(2)=V(2)-3*(D(1)-P(1));
         V(n)=V(n)-3*(P(2)-D(n));
         V(nx)=3*(P(2)-D(n))/H(n);
      case 'cu' % curvature specified spline
         B(1)=1;
         B(nx)=1;
         C(n)=0;
         A(n)=0;
         V(1)=P(1);
         V(2)=V(2)-H(1)*P(1);
         V(n)=V(n)-H(n)*P(2);
         V(nx)=P(2);
      end
      i=[2:nx 1:nx  1:n ];                          % row indices for A;B;C
      j=[1:n  1:nx  2:nx];                       % column indices for A;B;C
      ABC=sparse(i,j,[A;B;C],nx,nx,3*(nx+1));        % create sparse matrix

      switch pptype
      case {'sp' 'no' 'ex'}                        % poke in extra elements
         ABC(1,3)=H(1);
         ABC(nx,n-1)=H(n);
      end
      sflag=spparms('autommd');
      spparms('autommd',0);   % no reordering required
      m=ABC\V;                % find solution
      spparms('autommd',sflag);

      pp.coefs(:,4)=y(1:n); 
      pp.coefs(:,3)=D-H.*(2*m(1:n)+m(2:nx))/6;
      pp.coefs(:,2)=m(1:n)/2;
      pp.coefs(:,1)=diff(m)./(6*H);
      PPData.pp=pp;
      
      PPfun=@(x) PPfunction(PPData,x);
   end

otherwise
	 error('Unknown First Argument.')      
end
%--------------------------------------------------------------------------
function [xi,yi]=PPInverse(PPData,y)
% function called for inverse interpolation

if ischar(y) && strcmpi(y,'tag')
   xi=PPData.tag;
   return
end
if ~isnumeric(y) || numel(y)~=1
   error('Scalar Numerical Value Expected.')
end
pp=PPData.pp;
dxbr=diff(pp.breaks);
tol=100*eps;
xi=[];
pp.coefs(:,end)=pp.coefs(:,end)-y;  % shift all polys by y
for k=1:pp.pieces
   p=pp.coefs(k,:);     % k-th poly to investigate
   inz=find(p);         % index of nonzero elements of poly
   fnz=inz(1);          % first nonzero element
   lnz=inz(end);        % last nonzero element

   p=p(fnz+1:lnz)/p(fnz);        % strip leading,trailing zeros, make monic
   r=zeros(1,pp.order-lnz);      % roots at zero

   if (lnz-fnz)>1                   % add nonzero roots
      a=diag(ones(1,lnz-fnz-1),-1); % form companion matrix
      a(1,:)=-p;
      r = [r eig(a).'];             % find eignevalues to get roots
   end

   i=find(abs(imag(r))<tol & ... % find real roots with
          real(r)>=0 & ...       % nonnegative real parts that are
          real(r)<dxbr(k));      % less than the next breakpoint
   if ~isempty(i)
      xi=[xi; pp.breaks(k)+r(i)];
   end
end
if nargout==2
   yi=repmat(y,size(xi));
end
%--------------------------------------------------------------------------
function y=PPfunction(PPData,xx)
% function called when PPfun(x) is called.
pp = PPData.pp;
if ischar(xx)                                              % PPfun('string')
   switch lower(xx(1:2))
      case 'ta'
         y=PPData.tag;
      case 'pp'
         y=pp;
      case 'br'
         y=pp.breaks;
      case 'pi'
         y=pp.pieces;
      case 'or'
         y=pp.order;
      case 'pl'
         x=reshape(pp.breaks,1,[]);
         xlen=length(x);
         n=max(2,fix(500/xlen));
         xi=repmat(x(1:end-1),n,1);
         d=repmat((0:n-1)'/n,1,xlen-1);
         dx=repmat(diff(x),n,1);
         xi=xi+d.*dx;
         xi=[xi(:); x(xlen)];
         plot(xi,ppval(pp,xi))
      otherwise
         error('Invalid String Input.')
   end
elseif isnumeric(xx)                                             % PPfun(x)
   if numel(xx)==1   % scalar input
      idx=find(xx>=pp.breaks);
      if isempty(idx)               % extrapolate if necessary
         idx=1;
      elseif idx(end)>pp.pieces
         idx=pp.pieces;
      end
      xs=xx-pp.breaks(idx(end));    % local coordinates
      c=pp.coefs(idx(end),:);       % local polynomial
      if pp.order==4                % quick eval for cubic spline
         y=((c(1)*xs + c(2))*xs + c(3))*xs +c(4);
      else
         y=c(1);
         for i=2:pp.order           % apply Horner's method
            y=xs.*y+c(i);
         end
      end
   else              % array input
      xbr=pp.breaks;
      c=pp.coefs.';
      [rx,cx] = size(xx);
      xs=xx(:).';
      tosort=false;
      if any(diff(xs)<0)
         tosort=true;
         [xs,ix]=sort(xs);
      end
      % for each data point, compute its breakpoint interval
      [idx,idx]=histc(xs,xbr);
      idx(xs<xbr(1)|~isfinite(xs))=1; % extrapolate using first
      idx(xs>=xbr(end))=pp.pieces;    % and last polynomial

      xs=xs-xbr(idx);                 % local coordinates
      if pp.order==4                  % quick eval for cubic spline
         y=((c(1,idx).*xs + c(2,idx)).*xs + c(3,idx)).*xs +c(4,idx);
      else
         y=c(1,idx);
         for i=2:pp.order             % apply Horner's method
            y=xs.*y+c(i,idx);
         end
      end
      if tosort
         y(ix)=y;
      end
      y=reshape(y,rx,cx);
   end
else
   error('Numeric or String Input Expected.')
end
%--------------------------------------------------------------------------
