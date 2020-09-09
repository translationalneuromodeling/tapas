function [md5hash] = JavaMD5(filename)
%% [md5hash] = JavaMD5(filename)
% Use a java routine to calculate MD5 hash of a file
% Based on http://stackoverflow.com/a/304350/2531987
% Written by Marcin Konowalczyk
% Timmel Group @ Oxford University

narginchk(1,1);
assert(exist(filename,'file')==2,'JavaMD5:FileNotFoud','File not found');
try
    md = java.security.MessageDigest.getInstance('MD5');
    fid = fopen(filename); % <- this is much faster than java i/o stream
    digest = dec2hex(typecast(md.digest(fread(fid, inf, '*uint8')),'uint8'));
    fclose(fid);
    md5hash = lower(reshape(digest',1,[]));
catch me
    error('JavaMD5:HashError','Failed to calculate the MD5 hash');
end
end
