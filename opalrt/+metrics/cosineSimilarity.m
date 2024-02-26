function fullResult = cosineSimilarity(allParams)
    function result = cs(paramsA, paramsB)
        A = nn.ravel(paramsA);
        B = nn.ravel(paramsB);
        result = sum(A .* B) / (norm(A) * norm(B));
    endfunction

    numParams = rows(allParams);
    dists = zeros(numParams);
    for i = 1:numParams
        for j = 1:numParams
            if i != j
                dists(i, j) = cs(allParams(i), allParams(j));
            endif
        endfor
    endfor

    fullResult = abs(sum(sum(dists)) / (numParams^2 - numParams));
endfunction
