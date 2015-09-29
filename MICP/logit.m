% Logit function.
function a = logit(b)
    a = log(b./(1-b));
    a(real(a)~=a) = NaN;
end
