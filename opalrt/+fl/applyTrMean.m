function [updates, state] = applyTrMean(state, clientGrads)
    numClients = rows(clientGrads);
    rejectIdx = round(0.25 * numClients);
    updates = fl.aggregate(
        state,
        clientGrads,
        @(x,s,ln) mean(sort(x, 1, 'descend')(rejectIdx:numClients - rejectIdx), 1)
    );
endfunction
