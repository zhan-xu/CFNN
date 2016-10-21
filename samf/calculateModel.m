function [modelXf, modelAlphaf] = calculateModel(data,yf,lambda)
    xf = fft2(data);
    kf = linear_correlation(xf, xf);
    for i = 1:5
        modelAlphaf{i} = yf ./ (kf + lambda(i));   %equation for fast training
    end
    modelXf = xf;

end 