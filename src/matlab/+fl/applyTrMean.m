function [updates, state] = applyTrMean(state, clientGrads)
    numClients = rows(clientGrads);
    rejectIdx = round(0.25 * numClients);
    updates = fl.aggregate(
        clientGrads,
        @(x) mean(sort(x, 1)(rejectIdx:numClients - rejectIdx, :), 1));
endfunction
