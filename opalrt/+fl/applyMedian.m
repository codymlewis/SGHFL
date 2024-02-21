function [updates, state] = applyMedian(state, clientGrads)
    updates = fl.aggregate(state, clientGrads, @(x,s,ln) median(x, 1));
endfunction
