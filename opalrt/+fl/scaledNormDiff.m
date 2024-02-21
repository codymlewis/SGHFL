function newParams = scaledNormDiff(A, B, scale)
    fields = fieldnames(A);
    newParams = struct();
    for i = 1:numel(fields)
        newParams.(fields{i}) = struct(
            'W', scale * abs(A.(fields{i}).W - B.(fields{i}).W),
            'b', scale * abs(A.(fields{i}).b - B.(fields{i}).b)
        );
    endfor
endfunction
