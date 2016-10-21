function [img_files, pos, target_sz, ground_truth, video_path,datasetParam] = load_video_info(base_path, video,dataset)
%LOAD_VIDEO_INFO
%   This file is specialized to data set.


	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
    if ~strcmp(dataset,'RGBD')
        if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
            suffix = video(end-1:end);  %remember the suffix
            video = video(1:end-2);  %remove it from the video name
        else
            suffix = '';
        end
    end
	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
    
    switch dataset
        case 'RGBD'
            img_files=[];
            filename = [video_path video '.txt'];
        case 'TB-50'
            filename = [video_path 'groundtruth_rect' suffix '.txt'];
        case 'VOT14'
            filename = [video_path 'groundtruth.txt'];
        case 'VOT15'
            filename = [video_path 'groundtruth.txt'];
   end
    
	f = fopen(filename);
% 	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
    if f~=-1
	%the format is [x, y, width, height]
        switch dataset
            case 'RGBD' 
                try
                    ground_truth = textscan(f, '%f,%f,%f,%f,%f', 'ReturnOnError',false);  
                catch  % #ok, try different format (no commas)
                    frewind(f);
                    ground_truth = textscan(f, '%f %f %f %f %f');  
                end
            case 'TB-50'
                try
                    ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
                catch  % #ok, try different format (no commas)
                    frewind(f);
                    ground_truth = textscan(f, '%f %f %f %f');  
                end
            case 'VOT14'
                try
                    ground_truth = textscan(f, '%f,%f,%f,%f,%f,%f,%f,%f', 'ReturnOnError',false);  
                catch  % #ok, try different format (no commas)
                    frewind(f);
                    ground_truth = textscan(f, '%f %f %f %f');  
                end   
            case 'VOT15'
                try
                    ground_truth = textscan(f, '%f,%f,%f,%f,%f,%f,%f,%f', 'ReturnOnError',false);  
                catch  % #ok, try different format (no commas)
                    frewind(f);
                    ground_truth = textscan(f, '%f %f %f %f');  
                end   
        end
    else
        	filename = [video_path 'init.txt'];
            f = fopen(filename);
             ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
    end
	ground_truth = cat(2, ground_truth{:});
	fclose(f);
    
    if strcmp(dataset,'VOT14') || strcmp(dataset,'VOT15')
        x1 = min(ground_truth(:,1:2:end),[],2);
        x2 = max(ground_truth(:,1:2:end),[],2);
        y1 = min(ground_truth(:,2:2:end),[],2);
        y2 = max(ground_truth(:,2:2:end),[],2);

        ground_truth = [x1, y1, x2 - x1, y2 - y1];
    end
	
    
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
	
	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
	end
	
	
	%from now on, work in the subfolder where all the images are
    if strcmp(dataset,'TB-50')
        video_path = [video_path 'img/'];
    end
    
    


    %general case, just list all images
    img_files = dir([video_path '*.png']);
    if isempty(img_files),
        if strcmp(dataset,'RGBD')
            img_files = dir([video_path 'rgb/*.png']);
            imgs = {img_files.name};
            % sort
            n=numel(img_files);
            tmp=zeros(n,2);
            for i =1:n
                tmp(i,:)=sscanf(imgs{i},'r-%d-%d.png');
            end
            [~,idx] = sort(tmp(:,2));
            img_files = imgs(idx);
            
        else
            img_files = dir([video_path '*.jpg']);
            img_files = sort({img_files.name});
        end
        assert(~isempty(img_files), 'No image files to load.')
    end

    datasetParam={};
    datasetParam.dataset = dataset;
    datasetParam.numFrame = numel(img_files);
    if strcmp(dataset,'RGBD')
        datasetParam.depth = 1;
        datasetParam.frameInfo = load([video_path 'frames']);;
        video_path = [video_path 'rgb/'];
    else
        datasetParam.depth=0;
    end
end

