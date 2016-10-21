function results = run_NNSAMF(seq, res_path, bSaveImage)
    run matconvnet/matlab/vl_setupnn.m
    %get information (image file names, initial position, etc) from
    %the benchmark's workspace variables
    seq = evalin('base', 'subS');
    target_sz = seq.init_rect(1,[4,3]);
    pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
    img_files = seq.s_frames;
    video_path = [];
    [rects, time] = nsamf(video_path, img_files, pos, target_sz,[],seq.name);


    results.type = 'rect';
    results.res = rects;

    results.fps = size(rects,1)/time;

end