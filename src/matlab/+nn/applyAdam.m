function [updates, state] = applyAdam(state, grads, lr=0.01)
    updates = struct();
    state.t += 1;
    gradsFields = fieldnames(grads);
    for i = 1:numel(gradsFields)
        ## Update moments
        state.m.(gradsFields{i}).W = state.beta1 * state.m.(gradsFields{i}).W + (1 - state.beta1) * grads.(gradsFields{i}).W;
        state.m.(gradsFields{i}).b = state.beta1 * state.m.(gradsFields{i}).b + (1 - state.beta1) * grads.(gradsFields{i}).b;
        state.v.(gradsFields{i}).W = state.beta2 * state.v.(gradsFields{i}).W + (1 - state.beta2) * grads.(gradsFields{i}).W .^ 2;
        state.v.(gradsFields{i}).b = state.beta2 * state.v.(gradsFields{i}).b + (1 - state.beta2) * grads.(gradsFields{i}).b .^ 2;
        ## Bias correction
        mWHat = state.m.(gradsFields{i}).W ./ (1 - state.beta1 .^ state.t);
        mbHat = state.m.(gradsFields{i}).b ./ (1 - state.beta1 .^ state.t);
        vWHat = state.v.(gradsFields{i}).W ./ (1 - state.beta2 .^ state.t);
        vbHat = state.v.(gradsFields{i}).b ./ (1 - state.beta2 .^ state.t);
        ## Calculate update
        updates.(gradsFields{i}).W = lr .* mWHat ./ (sqrt(vWHat) + state.epsilon);
        updates.(gradsFields{i}).b = lr .* mbHat ./ (sqrt(vbHat) + state.epsilon);
    endfor
endfunction
