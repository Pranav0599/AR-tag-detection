import numpy as np
import cv2
import matplotlib.pyplot as  plt
import math


def calculate_homography(corners):
    #Calculating Homography matrix with corners of AR Tag and world frame


    # final_corners = np.array([[697, 355], [733, 221], [870, 259], [835, 393]])
    dimension = 200
    world_points = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
    A = []
    for index in range(0, len(corners)):
        x, y = corners[index][0], corners[index][1]
        u, v = world_points[index][0], world_points[index][1]

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

def warp_image(H, image, output_size):

    H_inv = np.linalg.inv(H)

    points_x = []
    points_y = []

    for row in range(output_size[0]):
        for col in range(output_size[1]):

            pos_vect = np.array([row, col, 1]).T
            pos_vect = H_inv.dot(pos_vect)
            pos_vect /= pos_vect[2] #normalizing with scaling factor

            points_x.append(int(pos_vect[0]))
            points_y.append(int(pos_vect[1]))

    img = []
    for x,y in zip(points_x, points_y):        
        img.append(image[y, x])

    img= np.reshape(img, output_size)

    return img

def warping(homography_mat, image):
    dimension = 200
    #Warping the the image to get just the AR tag and decoding it to get Tag ID
    # cropped_image =image[50:200, 50:200]  #cropping the image to isolate AR Tag
    # image = cv2.transpose(image)
    warped_image = np.zeros((dimension, dimension))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            new_vec = np.dot(homography_mat, [x, y, 1])
            # print(new_vec.shape)
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
    warped_image = cv2.transpose(warped_image)

    #Displaying Warped image - The Ar Tag 
    return warped_image

def image_fft(image):

    _,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(thresh, kernel, iterations=1) 

    dft = cv2.dft(np.float32(img_erosion), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:, :, 1]))

    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask = np.ones((rows,cols,2), np.uint8)
    r = 200
    center = [crow, ccol]
    x,y = np.ogrid[:rows, :cols]
    mask_area = (x-center[0])**2 + (y-center[1])**2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    fshift_mask_mag = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    image_back = cv2.idft(f_ishift)
    image_back = cv2.magnitude(image_back[:,:,0], image_back[:,:,1])

    return image_back, fshift_mask_mag

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
    # _,thresh = cv2.threshold(image, 249, 255, cv2.THRESH_BINARY)

    corners = cv2.goodFeaturesToTrack(image, 10 , 0.02 , 80)
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
        if distance < 200 and distance > 100:
            final_corners.append(corner)
    

    return final_corners

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

    dimension = 200

    #Printing teh ID and getting the oriantation of the AR Tag
    if np.array_equal(cropped_image[85, 85], corner_pixel):
        updated_coordinates = np.array([[0, 0], [dimension, 0], [dimension, dimension], [0, dimension]], dtype="float32")
        print("white square located at bottom right and ID is", list([block_3, block_4, block_2, block_1]))

    elif np.array_equal(cropped_image[15, 85],corner_pixel):
        updated_coordinates = np.array([[dimension, 0], [dimension, dimension], [0, dimension], [0, 0]], dtype="float32")
        print("white square located at top right and ID is" ,list([block_4, block_2, block_1, block_3])) 

    elif np.array_equal(cropped_image[15, 15],corner_pixel):
        updated_coordinates = np.array([[dimension, dimension], [0, dimension], [0, 0], [dimension, 0]], dtype="float32")
        print("white square located at top left and ID is" ,list([block_2, block_1, block_3, block_4])) 

    elif np.array_equal(cropped_image[85, 15],corner_pixel):
        updated_coordinates = np.array([[0, dimension], [0, 0], [dimension, 0], [dimension, dimension]], dtype="float32")
        print("white square located at bottom left and ID is" ,list([block_1, block_3, block_4, block_2])) 


if __name__ == "__main__":
    image = cv2.imread('AprilTag.png', 0)

    image_back, fshift_mask_mag = image_fft(image)
    final_corners = detect_corners(image_back)

    H = calculate_homography(final_corners)
    tag_image = warp_image(H, image, (200, 200))
    decode_tag(tag_image)

    plt.figure(figsize=(50,50)) 
    plt.subplot(2, 2, 1), plt.imshow(tag_image, cmap='gray') 
    plt.title('Isolated Tag Image, Tag ID and Oriantation printed in terminal'), plt.xticks([]), plt.yticks([])
    plt.show()
