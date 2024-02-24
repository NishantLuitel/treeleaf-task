import os
from pdf2image import convert_from_path
import cv2
import numpy as np


'''
This module when run outputs two images each with all the rectangles either rotated left or right to align them

'''

if __name__ == '__main__':

    image_name = 'rectangles.jpg'

    # If the image is not in the current directory convert the 2nd page of treeleaf task pdf to jpg
    if image_name not in os.listdir():
        print("Extracting image from treeleaf task pdf...")
        images = convert_from_path('treeleaf.pdf')
        images[1].save(image_name, 'JPEG')

    # Load the image
    i = cv2.imread(image_name)
    image = cv2.resize(i, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    edges = cv2.Canny(gray_image, 100, 200)

    # Find contours in the binary image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define an empty list to store rectangles
    rectangles = [[],[]] 

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        # Checking if it's a rectangle
        if len(approx) == 4 and cv2.contourArea(approx) > 500:
            x, y, w, h = cv2.boundingRect(approx)

            side1 = approx[1][0] - approx[0][0]
            side2 = approx[2][0] - approx[1][0]

            # Calculate the angle of the left side with respect to the x-axis
            angle_rad_1 = np.arctan2(side2[1], side2[0])
            angle_deg_1 = np.degrees(angle_rad_1)
            # Calculate the angle of the right side with respect to the x-axis
            angle_rad_2 = np.arctan2(side1[1], side1[0])
            angle_deg_2 = np.degrees(angle_rad_2)

            # Create a masked image with only one rectangle and everything inside it
            mask = np.zeros_like(image[:, :, 0]) 
            cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
            masked_edges = cv2.bitwise_and(gray_image, gray_image, mask=mask)

            #Make a white background instead of a black background around rectangles
            coords = approx.reshape(-1, 2)
            mask2 = np.ones_like(image[:,:,0]) * 255
            cv2.fillPoly(mask2, [coords], (0))
            masked_edges+=mask2

            #Calculate the centre to rotate about
            width,height = (2*x+ w),(2*y+h)
            center = width//2,height//2 

            # Generate rotation matrix
            rotation_matrix_1 = cv2.getRotationMatrix2D(center, angle_deg_1, 1.0)
            rotation_matrix_2 = cv2.getRotationMatrix2D(center, angle_deg_2, 1.0)
            # Perform the rotation
            rotated_image_1 = cv2.warpAffine(masked_edges, rotation_matrix_1, (image.shape[:2]),borderValue=(255))
            rotated_image_2 = cv2.warpAffine(masked_edges, rotation_matrix_2, (image.shape[:2]),borderValue=(255))
            
            rectangles[0].append(rotated_image_1)
            rectangles[1].append(rotated_image_2)

    final_image1=np.zeros_like(rectangles[0][0])
    for r in rectangles[0]:
        final_image1+=r

    cv2.imshow("Aligninng to left side", final_image1)
    cv2.imwrite('output/leftalign.png', final_image1)

    final_image2=np.zeros_like(rectangles[1][0])
    for r in rectangles[1]:
        final_image2+=r
    cv2.imshow("Aligninng to right side", final_image2)
    cv2.imwrite('output/rightalign.png', final_image2)
    cv2.waitKey(0)



