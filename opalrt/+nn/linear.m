function res = linear(params, x)
    res = sum(params.W .* x)' + params.b;
endfunction
