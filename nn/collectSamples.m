function [train_data, train_label] = collectSamples(img, train_data, train_label, model,param,net)
    %%
    pos = model.pos;
    searchSize = floor((model.target_sz * (1 + param.padding)));
    %randomly select positions for samples
    n_sample_reg=64; %number of samples from a frame
    samples_reg = zeros(n_sample_reg,2);
    samples_reg(:,1) = repmat(pos(1),n_sample_reg,1) + (floor(searchSize(1)/2)-5) * max(-1,min(1,0.7*randn(n_sample_reg,1))); %the random positions follow gaussian distribution
    samples_reg(:,2) = repmat(pos(2),n_sample_reg,1) + (floor(searchSize(2)/2)-5) * max(-1,min(1,0.7*randn(n_sample_reg,1)));
    samples_reg = round(samples_reg);
    
    train_data_curframe1 = [];
    train_label_curframe1 = [];
  
    for i = 1:n_sample_reg
        pos_new = samples_reg(i,:);
        patch = getPatch(img,pos_new,searchSize*param.window_scale,param.windowszBG);
        rawdata = prepareData(patch, param.features);
        dataconv = calculateFeatures(rawdata, param.features, param.cos_window_BG);
        dataconv = single(dataconv);
               
%         patch = getPatch(img,pos_new,searchSize,param.window_sz);
%         rawdata = prepareData(patch, param.features);
%         data = calculateFeatures(rawdata, param.features, param.cos_window);
%         dataconv = repmat(data,3,3,1);
%         dataconv = dataconv(size(data,1)-floor((size(data,1)-1)/2)+1:2*size(data,1)+floor(size(data,1)/2),...
%             size(data,2)-floor((size(data,2)-1)/2)+1:2*size(data,2)+floor(size(data,2)/2),:);
        train_data_curframe1(:,:,:,i)=dataconv;
        
        scale_factor = searchSize(2)/param.window_sz(2);
        pos_hog = (pos - pos_new)/scale_factor/+1;
%         pos_hog = (pos - pos_new)/param.features.cell_size+1;
        fsz = param.features.sz*param.features.cell_size;
        [rs, cs] = ndgrid((1:fsz(1)) - ceil(fsz(1)/2), (1:fsz(2)) - ceil(fsz(2)/2));
        output_sigma = sqrt(prod(param.features.cell_size*model.target_sz/scale_factor))*param.output_sigma_factor / param.features.cell_size;
        label = exp(-0.5 /output_sigma^2 * ((rs-pos_hog(1)).^2 + (cs-pos_hog(2)).^2));
%         label = repmat(label_center,3,3,1);
%         label_sz = size(label_center);
%         label = label(ceil(size(label,1)/2)+floor(label_sz(1)/2)-label_sz(1)+1: ...
%             ceil(size(label,1)/2)+floor(label_sz(1)/2), ...
%             ceil(size(label,2)/2)+floor(label_sz(2)/2)-label_sz(2)+1: ...
%             ceil(size(label,2)/2)+floor(label_sz(2)/2));
        label(label<1e-7)=0;
        train_label_curframe1(:,:,:,i)=label;

    end
    
    train_data_curframe2 = [];
    train_label_curframe2 = [];
    n_sample_bank = 40;
    n_sample_neg = 20;
    samples_neg = zeros(n_sample_bank,2);
    samples_neg(:,1) = repmat(pos(1),n_sample_bank,1) + (floor(searchSize(1)/2)-5) * max(-1,min(1,0.95*randn(n_sample_bank,1))); %the random positions follow gaussian distribution
    samples_neg(:,2) = repmat(pos(2),n_sample_bank,1) + (floor(searchSize(2)/2)-5) * max(-1,min(1,0.95*randn(n_sample_bank,1)));
    samples_neg = round(samples_neg);
    
    for i = 1:n_sample_bank
        pos_new = samples_neg(i,:);
        
        patch = getPatch(img,pos_new,searchSize*param.window_scale,param.windowszBG);
        rawdata = prepareData(patch, param.features);
        dataconv = calculateFeatures(rawdata, param.features, param.cos_window_BG);
        dataconv = single(dataconv);
        
%         patch = getPatch(img, pos_new, searchSize, param.window_sz);
%         rawdata = prepareData(patch, param.features);
%         data = calculateFeatures(rawdata, param.features, param.cos_window);
%         dataconv = repmat(data,3,3,1);
%         dataconv = dataconv(size(data,1)-floor((size(data,1)-1)/2)+1:2*size(data,1)+floor(size(data,1)/2),...
%             size(data,2)-floor((size(data,2)-1)/2)+1:2*size(data,2)+floor(size(data,2)/2),:);
        train_data_curframe2(:,:,:,i)=dataconv;

        scale_factor = searchSize(2)/param.window_sz(2);
        pos_hog = (pos - pos_new)/scale_factor/+1;
%         fsz = param.features.sz;
        [rs, cs] = ndgrid((1:fsz(1)) - ceil(fsz(1)/2), (1:fsz(2)) - ceil(fsz(2)/2));
        label = exp(-0.5 /param.output_sigma^2 * ((rs-pos_hog(1)).^2 + (cs-pos_hog(2)).^2));
        label(label<1e-7)=0;
        train_label_curframe2(:,:,:,i)=label;
    end
%     figure;
%     imshow(img_copy);
    batch = single(train_data_curframe2);
    if param.useGpu
        net = vl_simplenn_move(net, 'gpu') ;
        batch = gpuArray(batch);
    end
    res = vl_simplenn(net, batch);
    if param.useGpu
        finalmap = gather(res(end).x);
    else
        finalmap = res(end).x;
    end
    finalmap_reshape = reshape(finalmap,size(finalmap,1)*size(finalmap,2),n_sample_bank);
    response_max = max(finalmap_reshape);
    pos_max = zeros(n_sample_bank,2);
    for i = 1:n_sample_bank
        [pos_max(i,1),pos_max(i,2)]=find(finalmap(:,:,:,i)==response_max(i),1);
    end
    pos_max(:,1) = pos_max(:,1) - ceil(fsz(1)/2) - 1;
    pos_max(:,2) = pos_max(:,2) - ceil(fsz(2)/2) - 1;
    [~,index] = sort(pos_max(:,1).^2+pos_max(:,2).^2,'descend');
    train_data_refine = train_data_curframe2(:,:,:,index(1:n_sample_neg));
    train_label_refine = train_label_curframe2(:,:,:,index(1:n_sample_neg));
    
    
    
    
    %%
    
    train_data_curframe = cat(4,train_data_curframe1,train_data_refine);
    train_label_curframe = cat(4,train_label_curframe1,train_label_refine);
    
    train_data = cat(4,train_data,train_data_curframe); % put patch samples in the array
    train_label=cat(4,train_label,train_label_curframe); %put label samples in the array
end