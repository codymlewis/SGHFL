function newParams = subtractParams(A, B)
    fields = fieldnames(A);
    newParams = struct();
    for i = 1:numel(fields)
        newParams.(fields{i}) = struct(
            'W', A.(fields{i}).W - B.(fields{i}).W,
            'b', A.(fields{i}).b - B.(fields{i}).b
        );
    endfor
endfunction
