import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================

    output = np.zeros_like(image)
    output = image.copy()

    im_dim=image.shape
    flt_dim=imfilter.shape

    img_dim1=im_dim[0]
    img_dim2=im_dim[1]
    img_channels=im_dim[2] 
    flt_dim1=flt_dim[0]
    flt_dim2=flt_dim[1]

    pad_dim1=int((flt_dim1-1)/2)
    pad_dim2=int((flt_dim2-1)/2)
    pad_mat=np.zeros((img_dim1+2*pad_dim1,img_dim2+2*pad_dim2,3))
    pad_mat[pad_dim1: img_dim1 + pad_dim1, pad_dim2: img_dim2 + pad_dim2] = image

    filter_1D = imfilter.flatten()
    for d in range(img_channels):
        for i in range(img_dim1):
            for j in range(img_dim2):
                pat_mat_1D = pad_mat[i:i+flt_dim1,j:j+flt_dim2,d].flatten()
                output[i][j][d] = np.dot(filter_1D, pat_mat_1D)
    
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')

    return output