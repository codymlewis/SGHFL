function res = softmax(z)
    res = exp(z) / sum(exp(z));
endfunction
