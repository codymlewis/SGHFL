function [updates, state] = applyKrum(state, clientGrads)
    n = rows(clientGrads);
    clip = round(state.f * n);

    function distances = pairwiseDist(X)
        distances = zeros(rows(X));
        for i = 1:rows(X)
            for j = 1:rows(X)
                distances(i, j) = sqrt(sum(abs(X(i) - X(j)).^2));
            endfor
        endfor
    endfunction

    function winners = krum(X)
        distances = pairwiseDist(X) .^ 2;
        scores = sum(sort(distances, 2)(:, 2:n - clip), 2);
        [_, highScoresI] = sort(scores, "descend");
        winners = X(highScoresI(1:n - clip), :);
    endfunction

    updates = fl.aggregate(clientGrads, @(x) mean(krum(x), 1));
endfunction
