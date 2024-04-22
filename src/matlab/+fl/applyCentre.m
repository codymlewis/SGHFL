function [updates, state] = applyCentre(state, clientGrads)
    k = floor(rows(clientGrads) / 2) + 1;
    updates = fl.aggregate(
        clientGrads,
        @(x) mean(fl.kMeansClustering(fl.kMeansInit(x, k), x), 1));
endfunction
