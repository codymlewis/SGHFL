function [updates, state] = applyCentre(state, clientGrads)
    k = floor(rows(clientGrads) / 2) + 1;
    updates = fl.aggregate(state, clientGrads, @(x,s,ln) fl.kMeansInit(x, k));
endfunction
