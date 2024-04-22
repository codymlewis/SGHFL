function params = unravel(flatParams, params)
    fields = fieldnames(params);
    j = 1;
    for i = 1:numel(fields)
        wSize = size(params.(fields{i}).W);
        params.(fields{i}).W = reshape(flatParams(j:j + prod(wSize) - 1), wSize);
        j += prod(wSize);
        bSize = size(params.(fields{i}).b);
        params.(fields{i}).b = reshape(flatParams(j:j + prod(bSize) - 1), bSize);
        j += prod(bSize);
    endfor
endfunction
