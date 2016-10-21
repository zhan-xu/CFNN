% CFNN tracker

function rects = cfnn(video_path, img_files, pos, target_sz, datasetParam, name)


%get parameters
firstImg = imread([video_path img_files{1}]);
[model, param,net] = initCFNN(firstImg, pos, target_sz);

totalFrames = numel(img_files);

rects = zeros(totalFrames,4);
rects(1,:) = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

if isempty(datasetParam)
    param = displayManager(firstImg,rects(1,:),model,param,name);
end
if ~isempty(datasetParam)&&datasetParam.fout > 0
    fprintf(datasetParam.fout,'%.2f,%.2f,%.2f,%.2f\n', rects(1,1),rects(1,2),rects(1,3),rects(1,4));   
end

train_data = [];
train_label = [];
fprintf('frame %d: collecting samples...\n',1);
[train_data, train_label] = collectSamples(firstImg, train_data, train_label, model, param, net);
for frame=2:numel(img_files)
    origimg = imread([video_path img_files{frame}]);
    img = imresize(origimg,param.area_resize_factor);
    model = trackNN(img,model,param,net);
    if frame > 157
        update_interval = 20;
    else
        update_interval = 15;
    end
    if mod(frame,update_interval)==7
        fprintf('frame %d: collecting samples...\n',frame);
        [train_data, train_label] = collectSamples(img, train_data, train_label, model, param, net);
        fprintf('frame %d: tunnning net...\n',frame);       
        net = net_finetune(net,train_data,train_label);
        train_data = [];
        train_label = [];
    else
        fprintf('frame %d: collecting samples...\n',frame);
        [train_data, train_label] = collectSamples(img, train_data, train_label, model, param,net);
    end
    model.last_pos = model.pos;
    model.last_target_sz = model.target_sz;
    
    rect = [model.pos([2,1]) - model.target_sz([2,1])/2, model.target_sz([2,1])];
    rect = rect/param.area_resize_factor;
    rects(frame,:) = rect;
    if isempty(datasetParam)
        param = displayManager(origimg,rect,model,param,name);
    end
    if ~isempty(datasetParam)&&datasetParam.fout > 0
        fprintf(datasetParam.fout,'%.2f,%.2f,%.2f,%.2f\n', rects(frame,1),rects(frame,2),rects(frame,3),rects(frame,4));   
    end
end

end