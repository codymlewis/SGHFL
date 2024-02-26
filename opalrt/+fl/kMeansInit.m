## K-means++ initialisation
function centroids = kMeansInit(samples, k)
    function s = sample(x, n, weighted=false, w=[])
        selection = rand([rows(x), 1]);
        if weighted
            w = w ./ sum(w);
            bounds = cumsum(w);
            idx = max(repmat(1:rows(x)', [rows(x), 1]) .* (selection > bounds'), [], 2) + 1;
        endif
        s = x(idx(1:n), :);
    endfunction

    ## K-Means++ initialisation
    numSamples = rows(samples);
    centroids = [samples(randi(numSamples), :)];
    while rows(centroids) < k
        weights = zeros([rows(samples), rows(centroids)]);
        for i = 1:rows(centroids)
            norms = abs(samples - centroids(i, :)).^2;
            if size(samples)(2) > 1
                weights(:, i) = sum(norms, 2);
            else
                weights(:, i) = norms;
            endif
        endfor
        weights = sqrt(weights);
        p = min(weights, [], 2) .^ 2;
        centroids = cat(1, centroids, sample(samples, 1, true, p));
    endwhile
endfunction
