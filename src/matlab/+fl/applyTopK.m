function [updates, state] = applyTopK(state, clientGrads)
    function x = topk(x, k)
        K = round((1 - k) * numel(x));
        x = x .* (x >= sort(reshape(x, [], 1))(K));
    endfunction

    updates = fl.aggregate(clientGrads, @(x) topk(mean(x, 1), state.k));
endfunction
