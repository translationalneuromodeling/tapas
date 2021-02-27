function outp(address,byte)

global cogent;

%test for correct number of input arguments
if(nargin ~= 2)
    error('usage: outp(address,data)');
end

io64(cogent.io.ioObj,address,byte);
