% Safe sigmoid function. Return values below 1e-8 will be set to 1e-8.
function b = safesigm(a)
    b = 1./(1+exp(-a));
    b(b<1e-8) = 1e-8;
end
