function sa = as2sa(as)
    fields = fieldnames(as(1));
    sa = struct();
    C = rows(as);
    for i = 1:numel(fields)
        sa.(fields{i}).W = zeros([C, size(as(1).(fields{i}).W)]);
        sa.(fields{i}).b = zeros([C, size(as(1).(fields{i}).b)]);
        for c = 1:C
            sa.(fields{i}).W(c, :, :) = as(c).(fields{i}).W;
            sa.(fields{i}).b(c, :, :) = as(c).(fields{i}).b;
        endfor
    endfor
endfunction

