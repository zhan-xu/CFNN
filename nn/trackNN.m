function model = trackNN(img,model,param,net)

    pos=model.last_pos;
    target_sz=model.last_target_sz;
    responseR = zeros(3,size(param.search_size,2));
    finalmaps=cell(1,size(param.search_size,2));

    for i=1:size(param.search_size,2)
        tmp_sz = floor((target_sz * (1 + param.padding))* param.search_size(i));
        
%         patch = getPatch(img,pos,tmp_sz, param.window_sz);
%         rawdata = prepareData(patch, param.features);
%         data = calculateFeatures(rawdata, param.features, param.cos_window);
%        
%         datatmp = repmat(data,3,3,1);
%         dataNN_convNet = datatmp(size(data,1)-floor((size(data,1)-1)/2)+1:2*size(data,1)+floor(size(data,1)/2),...
%             size(data,2)-floor((size(data,2)-1)/2)+1:2*size(data,2)+floor(size(data,2)/2),:);
%         dataNN_convNet = single(dataNN_convNet);
        
        
        patch = getPatch(img,pos,tmp_sz*param.window_scale, param.windowszBG);
        rawdata = prepareData(patch, param.features);
        dataNN_convNet = calculateFeatures(rawdata, param.features, param.cos_window_BG);
        dataNN_convNet = single(dataNN_convNet);
        

        if param.useGpu
            net = vl_simplenn_move(net, 'gpu') ;
            dataNN_convNet = gpuArray(dataNN_convNet);
        end
        res = vl_simplenn(net, dataNN_convNet);
        % res2 = vl_simplenn(net, datatmp);
        if param.useGpu
            finalmap = gather(res(end).x);
        else
            finalmap = res(end).x;
        end

        fsz = size(finalmap);
        m =  max(finalmap(:));
        [vert_delta, horiz_delta] = find(finalmap ==m, 1);
        vert_delta = vert_delta - ceil(fsz(1)/2);
        horiz_delta = horiz_delta - ceil(fsz(2)/2);

        current_size = tmp_sz(2)/param.window_sz(2);
        tmpPos = pos + current_size* [vert_delta - 1, horiz_delta - 1];
        finalmaps{i}=finalmap;
        responseR(:,i) = [m tmpPos];          
    end
    
    [~, szid] = max(responseR(1,:));
    model.pos = responseR(2:3,szid)';
    d=finalmaps{szid};
    param.display{1}=90*d./sum(d(:));
    %update parameters
    model.target_sz = target_sz * param.search_size(szid);
end


