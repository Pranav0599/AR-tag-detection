## Importing Necessary Library
import numpy as np 
import cv2 
import scipy.fftpack 

#function to detect the wdges
def find_contours(thresh):
    
    # fast Fourier Transform of Image
    thresh = scipy.fft.fft2(thresh, axes = (0,1))
    fft_thresh_img_shifted = scipy.fft.fftshift(thresh)
    radius = 125
    rows, cols = thresh.shape
    center_x, center_y = int(rows / 2), int(cols / 2)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= np.square(radius)
    mask = np.ones((rows, cols)) 
    mask[mask_area] = 0
    fft_edge_img = fft_thresh_img_shifted * mask
    edge_image_shifted = scipy.fft.ifftshift(fft_edge_img)
    img_back_edge = scipy.fft.ifft2(edge_image_shifted)
    edged_image = np.abs(img_back_edge)

    return edged_image

## Function to Detect the Corners
def detect_corners(image_gray,image):
    

    kernel = np.ones((11,11),np.uint8)
    erosion = cv2.erode(image_gray,kernel,iterations = 1)
    image_dilated = cv2.dilate(erosion, kernel, iterations=1)
    corners = cv2.goodFeaturesToTrack(image_dilated, 10 , 0.1 , 100)
    corners = np.int0(corners)

    if len(corners) > 8:
        x = []
        y = []

        for i in range(0,len(corners)):
            a = corners[i]
            x.append(a[0,0])
            y.append(a[0,1])

        # Getting Sheet Corners
        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)

        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)

        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)

        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)
        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,255,0),2)
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,255,0),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,255,0),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,255,0),2)

        # Getting the Tag Corners
        Xmin_index = x.index(min(x))
        Xmin = x.pop(Xmin_index)
        Xmin_y = y.pop(Xmin_index)
        Xmax_index = x.index(max(x))
        Xmax = x.pop(Xmax_index)
        Xmax_y = y.pop(Xmax_index)
        Ymin_index = y.index(min(y))
        Ymin = y.pop(Ymin_index)
        Ymin_x = x.pop(Ymin_index)
        Ymax_index = y.index(max(y))
        Ymax = y.pop(Ymax_index)
        Ymax_x = x.pop(Ymax_index)
        # Drawing Line on the Tag
        image = cv2.line(image,(Xmin,Xmin_y),(Ymin_x,Ymin),(0,0,255),2)
        image = cv2.line(image,(Xmin,Xmin_y),(Ymax_x,Ymax),(0,0,255),2)
        image = cv2.line(image,(Ymax_x,Ymax),(Xmax,Xmax_y,),(0,0,255),2)
        image = cv2.line(image,(Ymin_x,Ymin),(Xmax,Xmax_y),(0,0,255),2)
        # Tag Corner Points
        tag_corner_points = np.array(([Ymin_x,Ymin],[Xmin,Xmin_y],[Ymax_x,Ymax],[Xmax,Xmax_y]))
        desired_tag_corner = np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]])
        return image, tag_corner_points, desired_tag_corner, Ymin, Ymax, Xmin, Xmax
    
    return image, None, None, None, None, None, None

def process_image(image_gray):
    
    # fast Fourier Transform of Image
    fft = scipy.fft.fft2(image_gray, axes = (0,1))
    fft_shifted = scipy.fft.fftshift(fft)
    kernel_x = 40
    kernel_y = 40
    cols, rows = image_gray.shape
    center_x, center_y = rows / 2, cols / 2
    rows = np.linspace(0, rows, rows)
    cols = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(rows, cols)
    Gmask = np.exp(-(np.square((X - center_x)/kernel_x) + np.square((Y - center_y)/kernel_y)))
    
    fft_image_blur = fft_shifted * Gmask

    # Getting back the original image after operation
    img_shifted_back = scipy.fft.ifftshift(fft_image_blur)
    img_back_blur = scipy.fft.ifft2(img_shifted_back)
    img_back_blur = np.abs(img_back_blur)
    img_blur = np.uint8(img_back_blur)

    return img_blur



## Homography
def calculate_homography(corners1, corners2):
    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0
    x = corners1[:, 0]
    y = corners1[:, 1]
    xp = corners2[:, 0]
    yp = corners2[:,1]
    nrows = 8
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]

    return H

## Function to calculate the Projection Matrix
def projection_matrix(h, K):  
    h1 = h[:,0]
    h2 = h[:,1]
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else: #else make it positive
        b = -1 * b_t  
    
    row1 = b[:, 0]
    row2 = b[:, 1]
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))

    P = np.matmul(K,Rt)  
    return(P,Rt,t)

## Function for Extracting the inner grid of the Tag
def tag_level_2(ref_tag_image):
    tag_size = 160
    ref_tag_image_gray = cv2.cvtColor(ref_tag_image, cv2.COLOR_BGR2GRAY)
    ref_tag_image_thresh = cv2.threshold(ref_tag_image_gray, 230 ,255,cv2.THRESH_BINARY)[1]
    ref_image_thresh_resized = cv2.resize(ref_tag_image_thresh, (tag_size, tag_size))
    grid_size = 8
    stride = int(tag_size/grid_size)
    grid = np.zeros((8,8))
    x = 0
    y = 0
    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            cell = ref_image_thresh_resized[y:y+stride, x:x+stride]
            if cell.mean() > 255//2:
                grid[i][j] = 255
            x = x + stride
        x = 0
        y = y + stride
    inner_grid = grid[2:6, 2:6]
    return inner_grid

def decode_tag(inner_grid):
    count = 0
    
    # Getting the oreintation of the Tag
    while not inner_grid[3,3] and count<4 :
        inner_grid = np.rot90(inner_grid,1)
        count+=1
    info_grid = inner_grid[1:3,1:3]
    info_grid_array = np.array((info_grid[0,0],info_grid[0,1],info_grid[1,1],info_grid[1,0]))
    tag_id = 0
    tag_id_bin = []
    for i in range(0,4):
        if(info_grid_array[i]) :
            tag_id = tag_id + 2**(i)
            tag_id_bin.append(1)
        else:
            tag_id_bin.append(0)

    return tag_id, tag_id_bin,count


## Warping function to get the TAG
def warp_perspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for i in range(maxHeight):
        for j in range(maxWidth):
            f = [i,j,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            warped[i][j] = img[int(yb)][int(xb)]
    return(warped)


## Placeing the image on the tag
def Place_image(image,testudo_img,corner_points,desired_tag_corner,Ymin,Ymax,Xmin,Xmax):
    # rows,cols,ch = image.shape
    H = calculate_homography( np.float32(corner_points),np.float32(desired_tag_corner))
    h_inv = np.linalg.inv(H)
    for a in range(0,tag.shape[1]):
        for b in range(0,tag.shape[0]):
            x, y, z = np.matmul(h_inv,[a,b,1])
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            image[int(yb)][int(xb)] = bilinear_interpolation(testudo_img, b, a)
    
    return image

def rotate_points(points):
    point_list = list(points.copy())
    top = point_list.pop(-1)
    point_list.insert(0, top)
    return np.array(point_list)

## Function to perform bilinear interpolation
def bilinear_interpolation(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id





# Main Function 
if __name__ == '__main__':
    
    ## Importing the Image and Video Files
    testudo_img = cv2.imread('testudo.png')
    # Defining the AR Tag size
    tag_size = 160
    testudo_img = cv2.resize(testudo_img, (tag_size,tag_size)) #Resize to the Tag Size
    ## Given Intrinsic Matrix Parameters
    K = np.array([[1346.1005953,0,932.163397529403],
       [ 0, 1355.93313621175,654.898679624155],
       [ 0, 0,1]])


    cap = cv2.VideoCapture('1tagvideo.mp4')

    ## Start getting Video Stream
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Streaming Stopped")
            break
        
        image = frame.copy()
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur = process_image(image_gray)
        ret,thresh = cv2.threshold(image_blur, 220 ,255,cv2.THRESH_BINARY)
        image_edge = find_contours(thresh)
        image_edge = np.uint8(image_edge)
        frame,corners,desired_tag_corner,Ymin,Ymax,Xmin,Xmax = detect_corners(image_gray,image)
     
        if corners is not None:
            
            H = calculate_homography( np.float32(corners),np.float32(desired_tag_corner))
            tag = warp_perspective( H, image,tag_size, tag_size)
            
            # Getting Tag pose and data
            tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
            ret, tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
            tag = cv2.cvtColor(tag,cv2.COLOR_GRAY2RGB)
            inner_grid = tag_level_2(tag) 
            tag_id, tag_id_bin, track_rotations = decode_tag(inner_grid) 
            for i in range(track_rotations):
                desired_tag_corner = rotate_points(desired_tag_corner)
            #Place the testudo image on the tag(superimposing)
            image = Place_image(image,testudo_img,corners,desired_tag_corner,Ymin,Ymax,Xmin,Xmax)
        try:
            cv2.imshow('Superimposed image', image)
        except:
            pass

        for corner in corners:
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 5)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


