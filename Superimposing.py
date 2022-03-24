import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def get_mean(x_pts, y_pts):

    mean = (np.mean(x_pts), np.mean(y_pts))

    while True:
        x_pts.append(mean[0])
        y_pts.append(mean[1])

        new_mean = (np.mean(x_pts), np.mean(y_pts))
        if new_mean == mean:
            mean = new_mean
            break

        mean = new_mean

    return mean


def detect_corners(image):
    _,thresh = cv2.threshold(image, 129, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    eroded_image = cv2.erode(thresh, kernel, iterations=1) 
    # image = eroded_image

    dft = cv2.dft(np.float32(eroded_image), flags=cv2.DFT_COMPLEX_OUTPUT) 
    dft_shift = np.fft.fftshift(dft) 
    r, c = image.shape
    m_radius = 220
    mask = np.ones((r, c, 2), np.uint8)
    m, n = int(r / 2), int(c / 2)
    for x in range(0,r):
        for y in range(0,c):
            if (x - m) ** 2 + (y - n) ** 2 < m_radius**2:
                    mask[x,y] = 0 
    mask_to_apply = mask
    
    fshift = dft_shift * mask_to_apply 
    f_ishift = np.fft.ifftshift(fshift) 

    fft_image = cv2.idft(f_ishift)
    fft_image = cv2.magnitude(fft_image[:, :, 0], fft_image[:, :, 1])

    corners = cv2.goodFeaturesToTrack(fft_image, 15 , 0.1 , 50)
    # corners = np.int0(corners)
    x_points = []
    y_points = []
    for i in corners:
        x,y = i.ravel()
        x_points.append(x)
        y_points.append(y)

    mean = get_mean(x_points, y_points)

    final_corners = []
    for corner in zip(x_points, y_points):
        distance = math.sqrt((corner[0] - mean[0])**2 + (corner[1] - mean[1])**2)
        if (distance < 210  and distance > 105):
            final_corners.append(corner)
    

    return final_corners

def sort_corners(corners):
    if len(corners) == 4:
        sort_by_x = sorted(corners, key=lambda x: x[0])
        bottom_points = [sort_by_x[0], sort_by_x[1]]
        top_points = [sort_by_x[2], sort_by_x[3]]

        top_points_sorted = sorted(top_points, key=lambda x: x[1])
        bottom_points_sorted = sorted(bottom_points, key=lambda x: x[1])

        t_l = top_points_sorted[0]
        t_r = top_points_sorted[1]
        b_l = bottom_points_sorted[0]
        b_r = bottom_points_sorted[1]

        return [t_l, t_r, b_r, b_l]


def calculate_homography(corners, tag_corners):
    print("gdgdgdgd")
    #Calculating Homography matrix with corners of required points

    # corners = corners
    
    world_points = tag_corners
    A = []
    for index in range(0, len(corners)):
        x, y = world_points[index][0], world_points[index][1]
        u, v = corners[index][0], corners[index][1]

        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)
    eig_values_1, eig_vects_1 = np.linalg.eig(np.matmul(A, A.T))
    eig_values_2, eig_vects_2 = np.linalg.eig(np.matmul(A.T, A))
    index_1 = eig_values_1.argsort()[::-1]
    eig_values_1 = eig_values_1[index_1]
    eig_vects_1 = eig_vects_1[:, index_1]
    index_2 = eig_values_2.argsort()[::-1]
    eig_values_2 = eig_values_2[index_2]
    eig_vects_2 = eig_vects_2[:, index_2]
    v_matrix = eig_vects_2
    homography_mat = np.zeros((A.shape[1], 1))
    for index in range(0, A.shape[1]):
        homography_mat[index, 0] = v_matrix[index, v_matrix.shape[1] - 1]
    homography_mat = homography_mat.reshape((3, 3))

    # scale the homography matrix by h[3][3] element
    for index1 in range(0, 3):
        for index2 in range(0, 3):
            homography_mat[index1][index2] = homography_mat[index1][index2] / homography_mat[2][2]

    return homography_mat

# def warp_image(H, image, output_size):

#     H_inv = np.linalg.inv(H)

#     points_x = []
#     points_y = []

#     for row in range(output_size[0]):
#         for col in range(output_size[1]):

#             pos_vect = np.array([row, col, 1]).T
#             pos_vect = H_inv.dot(pos_vect)
#             pos_vect /= pos_vect[2] #normalizing with scaling factor

#             points_x.append(int(pos_vect[0]))
#             points_y.append(int(pos_vect[1]))

#     img = []
#     for x,y in zip(points_x, points_y):        
#         img.append(image[y, x])

#     return img


def warp_image(H, image, dimension):
    # dimension = 200
    warped_image = np.zeros((dimension, dimension, 3))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            new_vec = np.dot(H, [x, y, 1])
            new_row, new_col, _ = (new_vec / new_vec[2]).astype(int)
            if new_row > 4 and new_row < (dimension - 4):
                if new_col > 4 and new_col < (dimension - 4):
                    warped_image[new_row, new_col] = image[x, y]
                    warped_image[new_row-1, new_col-1] = image[x, y]
                    warped_image[new_row-2, new_col-2] = image[x, y]
                    warped_image[new_row-3, new_col-3] = image[x, y]
                    warped_image[new_row+1, new_col+1] = image[x, y]
                    warped_image[new_row+2, new_col+2] = image[x, y]
                    warped_image[new_row+3, new_col+3] = image[x, y]

    warped_image = np.array(warped_image, dtype=np.uint8)

    return warped_image

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

def place_image(H, image, template_image):
    print("rsrsrsrsrsr")
    H_inv = np.linalg.inv(H)

    for a in range(0,template_image.shape[1]):
        for b in range(0,template_image.shape[0]):
            x, y, z = np.matmul(H_inv,[a,b,1])
            xb = np.clip(x/z,0,1919)
            yb = np.clip(y/z,0,1079)
            image[int(yb)][int(xb)] = bilinear_interpolation(template_image, b, a)

    

    # points_x = []
    # points_y = []
    # new_image = []

    # for row in range(template_image[0]):
    #     for col in range(template_image[1]):
    #         pos_vect1 = np.array([row, col, 1]).T
    #         pos_vect = np.dot(H_inv, pos_vect1)
    #         pos_vect /= pos_vect[2]
    #         points_x.append(int(pos_vect[0]))
    #         points_y.append(int(pos_vect[1]))
    #         new_image.append(list(template_image[row, col]))

    #     for index, (x,y) in enumerate(zip(points_x, points_y)):        
    #         image[y, x] = new_image[index]           

    return image



def decode_tag(warped_image):
    # bw = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(warped_image, 190, 255, cv2.THRESH_BINARY)
    cropped_image = bw[50:150, 50:150]


    #Decoding the AR Tag based on the oriantation and encoding pattern

    block_1 = cropped_image[37, 37]
    block_3 = cropped_image[62, 37]
    block_2 = cropped_image[37, 62]
    block_4 = cropped_image[62, 62]
    white = 255


    if np.array_equal(block_1, white):
        block_1 = 1
    else:
        block_1 = 0
    if np.array_equal(block_2, white):
        block_2 = 1
    else:
        block_2 = 0

    if np.array_equal(block_3, white):
        block_3 = 1
    else:
        block_3 = 0

    if np.array_equal(block_4, white):
        block_4 = 1
    else:
        block_4 = 0

    corner_pixel = 255

    # dimension = 200

    #Printing teh ID and getting the oriantation of the AR Tag
    if np.array_equal(cropped_image[85, 85], corner_pixel):
        oriantation = "up_right"
        id = list([block_3, block_4, block_2, block_1])
        print("white square located at bottom right and ID is", id )
        return oriantation

    elif np.array_equal(cropped_image[15, 85],corner_pixel):
        oriantation = "down_right"
        id = list([block_4, block_2, block_1, block_3])
        print("white square located at top right and ID is" ,id ) 
        return oriantation

    elif np.array_equal(cropped_image[15, 15],corner_pixel):
        oriantation = "down_left"
        id = list([block_2, block_1, block_3, block_4])
        print("white square located at top left and ID is" ,id ) 
        return oriantation

    elif np.array_equal(cropped_image[85, 15],corner_pixel):
        oriantation = "up_left"
        id = list([block_1, block_3, block_4, block_2])
        print("white square located at bottom left and ID is" ,id ) 
        return oriantation





if __name__ == "__main__":
    vid_cap = cv2.VideoCapture("1tagvideo.mp4")
    testudo_img1 = cv2.imread("testudo.png")
    tag_img_size = 200
    corners_of_projection = [(0,0), (tag_img_size-1,0), (0,tag_img_size-1), (tag_img_size-1, tag_img_size-1)]
    testudo_img = cv2.resize(testudo_img1, (tag_img_size,tag_img_size))
    testudo_img_corners = [(0,0), (0,testudo_img1.shape[1]),  (testudo_img1.shape[0], testudo_img1.shape[1]), (testudo_img1.shape[0],0)]
    counter = 0
    while True:
        _, frame = vid_cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # image = gray_image
        corners_of_tag = detect_corners(gray_image) 

        # if corners_of_tag is not None:
                        

        if corners_of_tag is not None:  
            print("sdfbedfbr")
        # try:
            print("yes")
            h_mat = calculate_homography(corners_of_tag, corners_of_projection)
            isolated_tag = warp_image(h_mat, gray_image, tag_img_size)
            # isilated_tag = cv2.cvtColor(isolated_tag, cv2.COLOR_BGR2GRAY)   
            # # warped_image = warp_image(h_mat, isilated_tag,dimension)
            # orientaion = decode_tag(isilated_tag) 

            # if orientaion == "up_left":        
            #     rotated_image = np.rot90(testudo_img, 3)
            # elif orientaion == "down_left":        
            #     rotated_image = np.rot90(testudo_img, 2)
            # elif orientaion == "up_right":
            #     rotated_image = np.rot90(testudo_img, 0)
            # elif orientaion == "down_right":
            #      rotated_image = np.rot90(testudo_img, 1)

            H_matrix_2 = calculate_homography( corners_of_tag,corners_of_projection)  
            # warped_img1 = warp_image(H_matrix_2, testudo_img, tag_img_size)
            place_image(H_matrix_2, frame, isolated_tag)
            # except:
            #     pass

            for corner in corners_of_tag:
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 3, (255, 0, 0), 5)


        cv2.imshow("Super Imposed", frame)

        if cv2.waitKey(20) == ord("q"):
            break
    
        # counter+= 1

    vid_cap.release()
    cv2.destroyAllWindows()
