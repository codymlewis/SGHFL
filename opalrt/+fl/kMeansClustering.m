## Lloyd's algorithm for clustering
function centroids = kMeansClustering(centroids, samples, numIterations=300, tol=0.0001)
    dists = zeros([rows(samples), rows(centroids)]);
    for iter = 1:numIterations
        for i = 1:rows(centroids)
            dists(:, i) = sqrt(sum(abs(samples - centroids(i, :)).^2, 2));
        endfor
        [errors, clusters] = min(dists, [], 2);
        for i = 1:rows(centroids)
            centroids(i, :) = mean(samples(clusters == i, :), 1);
        endfor
        if mean(errors) < tol
            break
        endif
    endfor
endfunction
