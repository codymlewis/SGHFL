function s = sample(x, n, weighted=false, w=[])
    selection = rand([rows(x), 1]);
    if weighted
        w = w ./ sum(w);
        bounds = cumsum(w);
        idx = max(repmat(1:rows(x)', [rows(x), 1]) .* (selection > bounds'), [], 2) + 1;
    endif
    s = x(idx(1:n), :);
endfunction
