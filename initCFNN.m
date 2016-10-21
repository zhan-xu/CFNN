function [model, param, net] = initCFNN(img, pos, target_sz)
    model = {};
    param = readParam();
    
    fixed_area = 80^2;
    param.area_resize_factor = sqrt(fixed_area/prod(target_sz));
    img = imresize(img,param.area_resize_factor);
    pos = floor(pos * param.area_resize_factor);
    target_sz = floor(target_sz * param.area_resize_factor);
    
    param.output_sigma = sqrt(prod(target_sz)) * param.output_sigma_factor / param.features.cell_size;
    if size(img,3)==1
        %display('Input image has single channel. We have shut off the color feature extraction.');
        %param.features.colorName = 0;
        repmat(img,[1,1,3]);
    end

    param.window_sz = floor(target_sz * (1 + param.padding));
%     param.window_sz = round(param.window_sz/param.features.cell_size)*param.features.cell_size;%make it a mutiple of 4
    if param.features.greyHoG
        param.features.sz=floor(param.window_sz / param.features.cell_size);
    else
        param.features.sz = param.window_sz;
        param.features.cell_size = 1;
    end
    param.cos_window = hann(param.features.sz(1)) *hann(param.features.sz(2))';
    
    
    param.features.szBG = 2*param.features.sz-1;
    param.windowszBG = param.features.cell_size*param.features.szBG;
    param.window_scale = param.windowszBG/param.window_sz;
    param.cos_window_BG = hann(param.features.szBG(1))*hann(param.features.szBG(2))';

   
    model.yf = fft2(gaussian_shaped_labels(param.output_sigma, param.features.sz));
    patch = getPatch(img,pos,param.window_sz, param.window_sz);
    rawdata = prepareData(patch, param.features);
    data = calculateFeatures(rawdata, param.features,param.cos_window);
    
    %create model
    [model.model_xf, model.model_alphaf] = calculateModel(data,model.yf,param.lambda);
    %%====== NN parameter ======
    numchn  = size(data,3);
    for n_lambda = 1:5
        Wc= real(ifft2(bsxfun(@times,conj(model.model_xf), model.model_alphaf{n_lambda})));
        Wc1=rot90(Wc,2);
        Wc_convNet(:,:,:,n_lambda) = Wc1;
    end
    bc = normrnd(0,0.0001,[1 5]);
%     Wf = 0.2*ones(1, 5)+normrnd(0,0.001,[1 5]);
    sz_Wf = [7,7];
    [rs,cs]=ndgrid((1:sz_Wf(1))-ceil(sz_Wf(1)/2),(1:sz_Wf(2))-ceil(sz_Wf(2)/2));
    Wf = exp(-0.5/1^2*(rs.^2+cs.^2));
    Wf = 0.2*repmat(Wf,1,1,5)+normrnd(0,0.001,[7,7,5]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Wc1=rot90(Wc,2);
%     Wc_convNet = reshape(Wc1,size(Wc1,1),size(Wc1,2),1,size(Wc1,3));
    net.layers = {};
    net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(Wc_convNet), single(bc)}}, ...
                           'learningRate', [.005 0.0005], ...
                           'filtersLearningRate', .005, ...
                           'biasesLearningRate', .0005, ...
                           'stride', 1, ...
                           'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'convt', ...
                           'weights', {{reshape(single(Wf),7,7,1,5), []}}, ...
                           'learningRate', [.005 0.0], ...
                           'filtersLearningRate', 0.005, ...
                           'biasesLearningRate', 0.0, ...
                           'stride', 1, ...
                           'pad', 0,...
                           'upsample',4,...
                           'crop',[3,0,3,0]) ;

    net = vl_simplenn_tidy(net) ;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param.features.numchn  = numchn;
        
        
    model.original_sz = target_sz;
    model.last_pos=pos;
    model.last_target_sz = target_sz;
    model.target_sz = target_sz;
    model.pos = pos;
    param.target_sz = target_sz;
    
end
