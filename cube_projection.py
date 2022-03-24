import cv2
import numpy as np
import matplotlib.pyplot as plt
from Superimposing import *


def P_Matrix(H_matrix, K_matrix):
    K_inv = np.linalg.inv(K_matrix)

    B_tilda = np.dot(K_inv, H_matrix)
    B_tilda_mod = np.linalg.norm(B_tilda)

    if B_tilda_mod < 0:
        B = -1  * B_tilda
    else:
        B =  B_tilda

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    lambda_ = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    lambda_ = 1 / lambda_

    r1 = lambda_ * b1
    r2 = lambda_ * b2
    r3 = np.cross(r1, r2)
    t = lambda_ * b3

    P = np.array([r1,r2, r3, t]).T
    P = np.dot(K_matrix, P)
    P = P / P[2,3]
    return P


def calculate_homography(corners):
    #Calculating Homography matrix with corners of AR Tag and world frame

    dimension = 200
    
    world_points = np.array([[0, 0], [dimension , 0], [dimension , dimension ], [0, dimension ]], dtype="float32")
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


def proj_coordinates(points, P_matrix):
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    z_pts = points[:, 2]

    points_3d_matrix = np.stack((x_pts, y_pts, z_pts, np.ones(x_pts.size)))
    transformed_points_matrix = P_matrix.dot(points_3d_matrix)
    transformed_points_matrix /= transformed_points_matrix[2,:]

    x_2d = np.int32(transformed_points_matrix[0,:])
    y_2d = np.int32(transformed_points_matrix[1,:])  

    projected_2d_points = np.dstack((x_2d, y_2d)).reshape(4,2)
    return projected_2d_points


def sort_coordinates(corners):
    if len(corners) == 4:
        x_sorted = sorted(corners, key=lambda x: x[0])
        bottom_points = [x_sorted[0], x_sorted[1]]
        top_points = [x_sorted[2], x_sorted[3]]
        top_points_sorted = sorted(top_points, key=lambda x: x[1])
        bottom_points_sorted = sorted(bottom_points, key=lambda x: x[1])
        top_l = top_points_sorted[0]
        top_r = top_points_sorted[1]
        bottom_l = bottom_points_sorted[0]
        bottom_r = bottom_points_sorted[1]

        return [top_l, top_r, bottom_r, bottom_l]



def proj_cube(image, bottom_points, top_points):

    # sort_by_x = sorted(bottom_points, key=lambda x: x[0])
    # bottom_points = [sort_by_x[0], sort_by_x[1]]
    # top_points = [sort_by_x[2], sort_by_x[3]]
    # top_points_sorted = sorted(top_points, key=lambda x: x[1])
    # bottom_points_sorted = sorted(bottom_points, key=lambda x: x[1])
    # t_l = top_points_sorted[0]
    # t_r = top_points_sorted[1]
    # b_l = bottom_points_sorted[0]
    # b_r = bottom_points_sorted[1]

    bottom_points = sort_coordinates(bottom_points)
    bottom_points = np.array(bottom_points, np.int32)
    image = cv2.polylines(image, [bottom_points],True, 150, 2)

    top_points = sort_coordinates(top_points)
    top_points = np.array(top_points, np.int32)
    image = cv2.polylines(image, [top_points],True, 150, 2)

    for i in range(0, bottom_points.shape[0]):
        color = 150
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), color, 3)

    return image




if __name__ == "__main__":
    vid_cap = cv2.VideoCapture("1tagvideo.mp4")
    K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])
    desired_tag_img_size = 300
    desired_corners = [(0,0), (desired_tag_img_size,0), (0,desired_tag_img_size), (desired_tag_img_size, desired_tag_img_size)]
    cube_height_ = 200
    cube_height_1 = np.array([-(cube_height_), -(cube_height_), -(cube_height_), -(cube_height_)]).reshape(-1,1)
    cube_top_corners = np.concatenate((desired_corners, cube_height_1), axis = 1)


    frame_counter = 0
    while True:
        _, frame = vid_cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_counter % 5 == 0:
            # image = gray_image
            tag_corners = detect_corners(gray_image)

            if len(tag_corners) == 4:
                Hdt = calculate_homography(np.float32(tag_corners))
                P = P_Matrix(Hdt, K)
                cube_top_projected_corners = proj_coordinates(cube_top_corners, P)
                proj_cube(frame, tag_corners, cube_top_projected_corners)
                # x1,y1,z1 = np.matmul(P,[0,0,0,1])
                # x2,y2,z2 = np.matmul(P,[0,200,0,1])
                # x3,y3,z3 = np.matmul(P,[200,0,0,1])
                # x4,y4,z4 = np.matmul(P,[200,200,0,1])
                # x5,y5,z5 = np.matmul(P,[0,0,-200,1])
                # x6,y6,z6 = np.matmul(P,[0,200,-200,1])
                # x7,y7,z7 = np.matmul(P,[200,0,-200,1])
                # x8,y8,z8 = np.matmul(P,[200,200,-200,1])


                # cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
                # cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
                # cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
                # cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

                # cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
                # cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
                # cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
                # cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

                # cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
                # cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
                # cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
                # cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)

            cv2.imshow("canny_edge", frame)

            if cv2.waitKey(5) == ord("q"):
                break
                
    vid_cap.release()
    cv2.destroyAllWindows()