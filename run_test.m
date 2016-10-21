% Used to test tracker
%

run matconvnet/matlab/vl_setupnn.m
%path to the videos (you'll be able to choose one with the GUI).
datasets={struct('name','TB-50','basePath','/home/enderhsu/matlabWorkspace/Data')};

%ask the user for the video, then call self with that video name.
[video, base_path, dataset] = choose_video(datasets);

%get image file names, initial state, and ground truth for evaluation
[img_files, pos, target_sz, ground_truth, video_path,datasetParam] = load_video_info(base_path, video,dataset);

%call tracker function
rects = cfnn(video_path, img_files, pos, target_sz,[],video);

%calculate and show precision plot, as well as frames-per-second
if ~strcmp(dataset,'RGBD')
    precisions = precision_plot(rects, ground_truth, video, 1);
end

fprintf('%12s - Precision (20px):% 1.3f\n', video, precisions(20))
