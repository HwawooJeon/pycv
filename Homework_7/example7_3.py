
def get_HOGimg( Input_image, nCell_size, nBlock_size, edge_norient = 8):
    "Tiis function is to get histogram of gradient"

    import math
    import sys
    import numpy as np
    from scipy import sqrt, pi, arctan2, cos, sin
    from scipy.ndimage import uniform_filter
    import matplotlib.pyplot as plt



    ## Function Name: get_HOGimg ******************************************
    #
    # Description: Function of HOG
    # Author: Hwawoo Jeon (feelgood88@nate.com)
    #
    # Input
    #  - Input_image: grayscale ndarray
    #  - nCell_size: cell size (intger)
    #  - nBlock_size: block size (intger)
    #
    #***********************************************************************

    Input_image = np.atleast_2d(Input_image)
    
    #gamma correction
    image = sqrt(image)

    #Gamma Correction
    encoded = ((original / 255) ^ (1 / 0.5)) * 255
    
    #set value
    [ mWidth, mHeight] = Input_image.shape;

    g_width = np.zeros(Input_image.shape);
    g_height = np.zeros(Input_image.shape);
    g_width[:, :-1] = np.diff(Input_image, n=1, axis=1)
    g_height[:-1, :] = np.diff(Input_image, n=1, axis=0)

    m_magnitude = sqrt(g_width ** 2 + g_height ** 2)
    m_orientation = arctan2(g_height, (g_width + 1e-15)) * (180 / pi) + 90

    m_cell_width = nCell_size
    m_cell_height = nCell_size

    m_block_width = nBlock_size
    m_block_height = nBlock_size

    m_nCells_width = int(np.floor(mWidth // m_cell_width))  # number of cells in x : intiger
    m_nCells_height = int(np.floor(mHeight // m_cell_height))  # number of cells in y : intiger

    #Calculate Orientation Hist
    mOrientation_Hist = np.zeros( (m_nCells_width, m_nCells_height, edge_norient) )

    for n in range(edge_norient):
        
        temp = np.where(m_orientation < 180 / edge_norient * (n + 1), m_orientation, 0)
        temp = np.where(m_orientation >= 180 / edge_norient * n, temp, 0)
        # select magnitudes for those orientations
        #true or false
        cond2 = temp < 0
        temp_mag = np.where(cond2, m_magnitude, 0)

        mOrientation_Hist[:,:,i] = uniform_filter(temp_mag, size=(m_cell_height, m_cell_width))[m_cell_height/2::m_cell_height, m_cell_width/2::m_cell_width]

    #For display hog image

    radius = min(m_cell_width, m_block_height) // 2 - 1
    hog_image = None
    hog_image = np.zeros((mHeight, mWidth), dtype=float)
    
    for x in range(n_cellsx):
        for y in range(n_cellsy):
            for o in range(orientations):
                centre = tuple([y * m_cell_height + m_cell_height // 2, x * m_cell_width + m_cell_width // 2])
                dx = radius * cos(float(o) / m_orientation * np.pi)
                dy = radius * sin(float(o) / m_orientation * np.pi)
                rr, cc = draw.bresenham(centre[0] - dx, centre[1] - dy, centre[0] + dx, centre[1] + dy)
                hog_image[rr, cc] += mOrientation_Hist[y, x, o]

    plt.imshow(hog_image)

    n_blocksx = (m_nCells_width - m_block_width) + 1
    n_blocksy = (m_nCells_height - m_block_height) + 1
    mResult_Image = np.zeros((n_blocksy, n_blocksx, m_block_height, m_block_width, m_orientation))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = mOrientation_Hist[y:y + m_block_height, x:x + m_block_width, :]
            eps = 1e-5
            mResult_Image[y, x, :] = block / sqrt(block.sum() ** 2 + eps)
            
    return mResult_Image
