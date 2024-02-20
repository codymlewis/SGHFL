function error = meanAbsoluteError(truth, predictions)
    error = mean(abs(truth - predictions), 'all');
endfunction
