function score = r2score(truth, predictions)
    YBar = mean(truth);
    SSRes = sum(sum((truth - predictions) .^ 2));
    SSTot = sum(sum((truth - YBar) .^ 2));
    score = 1 - (SSRes / SSTot);
endfunction
