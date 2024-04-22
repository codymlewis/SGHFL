function [updates, state] = applyFedProx(state, clientGrads)
    [updates, flatUpdates] = fl.aggregate(
        clientGrads,
        @(x) mean(x, 1) - state.mu * abs(state.prevParams - state.currentParams));
    state.iteration += 1;
    state.prevParams = state.currentParams;
    state.currentParams -= flatUpdates;
endfunction
