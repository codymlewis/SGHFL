function flatCParams = flattenClientParams(clientParams)
    flatCParams = [];
    for i = 1:rows(clientParams)
        flatCParams = cat(1, flatCParams, nn.ravel(clientParams(i))');
    endfor
endfunction
