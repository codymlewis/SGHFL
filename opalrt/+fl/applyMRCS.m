function [updates, state] = applyMRCS(state, clientGrads)
    function sims = cs(A, X)
        sims = zeros([rows(X), 1]);
        for i = 1:rows(X)
            sims(i) = sum(A .* X(i)) / (norm(A) * norm(X(i)));
        endfor
    endfunction

    function aggGrads = mrcs(X)
        pVals = ones([rows(X), 1]);
        if any(any(state.momentum != 0))
            sims = cs(state.momentum, X);
            pVals = max(0, sims);
        endif
        if sum(pVals) == 0
            pVals = ones([rows(X), 1]);
        endif
        pVals = pVals ./ sum(pVals);
        aggGrads = sum(pVals .* X, 1);
        momentum = (1 - state.mu) * state.momentum + state.mu * aggGrads;
    endfunction

    function result = polyakAverage(oldValue, newValue, mu)
        result = (1 - mu) .* oldValue + mu * newValue;
    endfunction

    aggGrads = fl.aggregate(clientGrads, @(x) mrcs(x));
    state.momentum = polyakAverage(state.momentum, nn.ravel(aggGrads), state.mu);
    updates = nn.unravel(state.momentum, clientGrads(1));
endfunction
