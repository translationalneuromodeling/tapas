function config_io

global cogent;

%create IO64 interface object
cogent.io.ioObj = io64();

%install the inpoutx64.dll driver
%status = 0 if installation successful
cogent.io.status = io64(cogent.io.ioObj);
if(cogent.io.status ~= 0)
    disp('inp/outp installation failed!')
end
