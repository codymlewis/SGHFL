function [updates, state] = applyFedavg(state, clientGrads)
    updates = fl.aggregate(state, clientGrads, @(x,s,ln) mean(x, 1));
endfunction
