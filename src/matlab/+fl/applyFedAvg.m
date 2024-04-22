function [updates, state] = applyFedAvg(state, clientGrads)
    updates = fl.aggregate(clientGrads, @(x) mean(x, 1));
endfunction
