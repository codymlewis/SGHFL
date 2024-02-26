function [updates, state] = applyKickbackMomentum(state, clientGrads)
    state.momentum = state.mu1 * state.momentum + (state.prevParams - state.currentParams);
    [updates, flatUpdates] = fl.aggregate(clientGrads, @(x) state.mu2 * state.momentum + mean(x, 1));
    state.prevParams = state.currentParams;
    state.currentParams -= flatUpdates;
endfunction
