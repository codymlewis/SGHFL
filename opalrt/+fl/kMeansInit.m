function centroids = kMeansInit(samples, k)
    ## K-Means++ initialisation
    numSamples = rows(samples);
    centroids = [samples(randi(numSamples), :)];
    while rows(centroids) < k
        weights = zeros([rows(samples), rows(centroids)]);
        for i = 1:rows(centroids)
            weights(:, i) = abs(samples - centroids(i, :)).^2;
            if size(weights(:, i))(2) > 1
                weights(:, i) = sum(weights(:, i), 2);
            endif
        endfor
        weights = sqrt(weights);
        p = min(weights, [], 2) .^ 2;
        centroids = cat(1, centroids, utils.sample(samples, 1, true, p));
    endwhile
endfunction
