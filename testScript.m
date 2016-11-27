[m,n] = size(incidenceFull);
count = 0;
for k = 1:m
    u = find(incidenceFull(k,:));
    for a = 1:size(u,2)
        if (u(a)<=m)
            if incidenceFull(u(a),k) == 1
                count =  count + 1;
            end
        end
        
    end
end
count
size(find(incidenceFull(:)))