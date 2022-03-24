import numpy as np
import cv2
import matplotlib.pyplot as  plt
import math





def image_fft(image):

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
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

def detect_corners(image):
    _,thresh = cv2.threshold(image, 249, 255, cv2.THRESH_BINARY)

    corners = cv2.goodFeaturesToTrack(thresh, 11 , 0.04 , 30)
    corners = np.int0(corners)
    x_points = []
    y_points = []
    for i in corners:
        x,y = i.ravel()
        x_points.append(x)
        y_points.append(y)


    final_corners = []
    for i in x_points:
        for j in y_points:
            if x_points.index(i) == y_points.index(j):
                final_corners.append([i,j])
    x = 800
    y = 290

    for corner in final_corners:
        distance = math.sqrt((corner[0] - x)**2 + (corner[1] - y)**2)
        if distance > 200 or distance < 100:
            final_corners.pop(final_corners.index(corner))

    final_corners.append((x, y))
    for i in final_corners:
        x, y = i[0], i[1]
        cv2.circle(image,(x,y),5,(0,0,255),-1)

    return image





if __name__ == "__main__":
    image = cv2.imread('AprilTag.png', 0)

    image_back, fshift_mask_mag = image_fft(image)
    corners_image = detect_corners(image)

    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(image, cmap = 'gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(fshift_mask_mag, cmap = 'gray')
    ax2.title.set_text('FFT of Image')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(corners_image, cmap = 'gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(image_back, cmap = 'gray')
    ax4.title.set_text('Image after inverse FFT')
    plt.show()

