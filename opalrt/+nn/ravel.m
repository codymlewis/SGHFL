function flatParams = ravel(params)
    fields = fieldnames(params);
    flatParams = [];
    for i = 1:numel(fields)
        flatParams = cat(1, flatParams, reshape(params.(fields{i}).W, [], 1), reshape(params.(fields{i}).b, [], 1));
    endfor
endfunction
