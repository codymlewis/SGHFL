function [updates, state] = applyMedian(state, clientGrads)
    updates = fl.aggregate(clientGrads, @(x) median(x, 1));
endfunction
