function net = net_finetune(net,train_data,train_label)

opts.useGpu = true;%true
opts.learningRate = 0.0003;%0.0006
opts.weightDecay = 0.001 ;
opts.momentum = 0.9 ;
opts.batchSize = 84;
n_data = size(train_data,4);
opts.maxiter = 20;
opts.weightDecay=0.01;

time_disp = 0;

train_data = single(train_data);
train_label = single(train_label);

tic
if ~isfield(net,'lr')
    lr = opts.learningRate;
else
    lr = net.lr;
end

train_samples = [];
train_cnt = 0;
remain = opts.batchSize*opts.maxiter;
while(remain>0)
    if(train_cnt==0)
        train_list = randperm(n_data)';
    end
    train_samples = cat(1,train_samples,...
        train_list(train_cnt+1:min(end,train_cnt+remain)));
    train_cnt = min(length(train_list),train_cnt+remain);
    train_cnt = mod(train_cnt,length(train_list));
    remain = opts.batchSize*opts.maxiter-length(train_samples);
end

if time_disp == 1
    time1 = toc;
    fprintf('time1 = %d\n',time1);
    tic
end

net.layers{end+1} = struct('type', 'pdist');
net.layers{end}.name = 'pdist';
net.layers{end}.p = 2;
net.layers{end}.noRoot=true;
net.layers{end}.aggregate=true;
net.layers{end}.epsilon = 1e-6;
net.layers{end}.instanceWeights = [];

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type, 'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.weights{1}), 'double') ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.weights{2}), 'double') ;      
        net.layers{i}.filtersWeightDecay = ones(1,1,'double');
        net.layers{i}.biasesWeightDecay = ones(1,1,'double');
    end
end

if opts.useGpu 
    net = vl_simplenn_move(net, 'gpu') ;
else
    net = vl_simplenn_move(net, 'cpu') ;
end


cost_last = 0;
% lr_last = lr;
t=1;
net_last = net;
cost_array=[];
repeat_time = 0;
cost_original  = 0;
lr_changes = 0;

if time_disp == 1
    time2 = toc;
    fprintf('time2 = %d\n',time2);
    tic
end

while t <= opts.maxiter
    %prepare data and labels for mini-batch
    lr = min(lr,opts.learningRate);
    batch = train_data(:,:,:,train_samples((t-1)*opts.batchSize+1:t*opts.batchSize));
    labels = train_label(:,:,:,train_samples((t-1)*opts.batchSize+1:t*opts.batchSize));
    net.layers{end}.class = labels;
    %put matrices into Gpu
    if opts.useGpu
        batch = gpuArray(batch);
        one = gpuArray(1.0);
    else
        one = 1.0;
    end
    
    
    res = [];
    
    if time_disp == 1
        time3 = toc;
        fprintf('time3 = %d\n',time3);
        tic
    end
    
    res = vl_simplenn(net, batch, one, res);
    
    if time_disp == 1
        time4 = toc;
        fprintf('time4 = %d\n',time4);
        tic
    end
    
    %calculate cost
    cost = sum(gather(res(end).x))/opts.batchSize;% + lambda.* (sum(Wc(:).^2)+sum(Wf(:).^2)); 
    cost1 = cost;
    for l = 1:numel(net.layers)
        switch net.layers{l}.type
            case 'conv'
                cost1 = cost1 + opts.weightDecay.*(sum(net.layers{l}.weights{1}(:).^2)+sum(net.layers{l}.weights{2}(:).^2));
%                 cost1 = cost1 + opts.weightDecay.*(sum(net.layers{l}.weights{1}(:).^2));
        end
    end
    cost = cost * .5;
    cost1 = cost1 * .5;
    
    if t == 1
        cost_original = cost;
    end
    
    if cost/cost_last >= 1.5 && t~=1 && repeat_time==0
        net = net_last;      
        fprintf('this step gets too far!\n'); 
         lr = lr/2;
%        lr = lr/1;
        lr_changes = lr_changes + 1;
        if lr_changes == 4
            break;
        end
        repeat_time = repeat_time + 1;
        continue;
    else
        repeat_time = 0;
        cost_last = cost;
        cost_array = [cost_array, cost];
        % print information
        fprintf('iter = %d, lr = %f, cost1 = %.3f, cost = %.3f\n', t,lr,cost,cost1);
    end
    
    if size(cost_array,2)>=2 && cost_array(end)>cost_original*2
        net = net_last;
        break;
    end
    
    net_last = net;
    
    if time_disp == 1
        time5 = toc;
        fprintf('time5 = %d\n',time5);
        tic
    end
    
    % gradient step
    for l=1:numel(net.layers)
        if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
        
        net.layers{l}.filtersMomentum = ...
            opts.momentum * net.layers{l}.filtersMomentum ...
            - (lr * net.layers{l}.filtersLearningRate) * ...
            (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.weights{1} ...
            - (lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1} ;
%         p1 = gather(opts.momentum * net.layers{l}.filtersMomentum);
%         p2 = gather((lr * net.layers{l}.filtersLearningRate) *(opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.weights{1});
%         p3 = gather((lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1});
%         p4 = gather(net.layers{l}.filtersMomentum);
%         p5 = gather(net.layers{l}.weights{1});
        net.layers{l}.biasesMomentum = ...
            opts.momentum * net.layers{l}.biasesMomentum ...
            - (lr * net.layers{l}.biasesLearningRate) * ....
            (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.weights{2} ...
            - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;

        net.layers{l}.weights{1} = net.layers{l}.weights{1} + net.layers{l}.filtersMomentum ;
        net.layers{l}.weights{2} = net.layers{l}.weights{2} + net.layers{l}.biasesMomentum ;
    end
    
    lr = lr*1.07;
    %lr = lr*1.0;
    t = t+1;
    if size(cost_array,2)>14
        rate = mean(cost_array(end-5:end))/mean(cost_array(end-11:end-6));
        fprintf('rate = %d\n',rate);
        if rate>0.98
            fprintf('early stop.\n');
            break;
        end
    end
    
    if time_disp == 1
        time6 = toc;
        fprintf('time6 = %d\n',time6);
    end
    
end

net.layers = net.layers(1:end-1);
net.lr = lr;