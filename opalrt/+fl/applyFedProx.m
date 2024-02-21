function [updates, state] = applyFedProx(state, clientGrads)
    updates = fl.aggregate(state, clientGrads, @(x,s,ln) mean(x, 1));
    updates = nn.subtractParams(updates, fl.scaledNormDiff(state.prevParams, state.currentParams, 0.0001));
    state.iteration += 1;
    state.prevParams = state.currentParams;
    state.currentParams = nn.applyUpdates(state.currentParams, updates);
endfunction
