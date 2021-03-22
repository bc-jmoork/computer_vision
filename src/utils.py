import numpy as np

def convolution(image, kernel, padding = True, bias=0.):
    """
    Simple 2D convolution of image with or without padding
    You can add a constant bias if necessary
    """
    K = kernel.shape[0]
    pad = K//2
    M, N = image.shape
    conv_image = np.zeros_like(image, dtype='int')
    
    # save padded image
    if padding:
        image_padded = np.zeros(((image.shape[0] + K - 1), (image.shape[1] + K - 1)), dtype='int')
        image_padded[pad:-pad,pad:-pad] = image
    else:
        M = M - K + 1
        N = N - K + 1
        image_padded = image
    
    for i in range(M):
        for j in range(N):
            sum = np.sum(image_padded[i:i+K, j:j+K]*kernel)
            conv_image[i][j] = sum + bias
 
    return conv_image


def fft_image(norm_image):
    '''This function takes in a normalized, grayscale image
       and returns a frequency spectrum transform of that image. '''
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    
    return frequency_tx

