# import modules
import os
from pdf2image import convert_from_path
import cv2
import numpy as np

def find_rectangles_with_inside_line_length(image,view_all_contours=False):

  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply edge detection
  edges = cv2.Canny(gray, 100, 200)

  # Find contours (only outer rectangle but not the lines)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  rectangles = []

  # Loop through contours
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

    # Checking if it's a rectangle
    if len(approx) == 4 and cv2.contourArea(approx) > 500:
      x, y, w, h = cv2.boundingRect(approx)

      # Create a masked image with only one rectangle and everything inside it
      mask = np.zeros_like(image[:, :, 0])
      cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
      masked_edges = edges & mask
    
      # Find the contours(This time using RETR_TREE to find nested contour)
      line_contours, _ = cv2.findContours(masked_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contour_img = np.zeros_like(image)

    
      largest_line_length = 20000
      # Find the smallest contour in the line_contours because the larger contours are the rectangles themselves
      for line_cnt in line_contours:
        line_length = cv2.arcLength(line_cnt, False)
        if line_length < largest_line_length:
          largest_line_length = line_length
        contour_img = np.zeros_like(image)

        if view_all_contours:
            # Draw contours on the black image
            cv2.drawContours(contour_img, [line_cnt], -1, (0, 255, 0), 2)

            # Display the image with contours
            cv2.imshow('Image with Contours', contour_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
      rectangles.append((x, y, w, h, largest_line_length))

  #  Sort rectangles by line length in descending order
  rectangles.sort(key=lambda x: x[4])
  return rectangles

if __name__ == '__main__':
  
# The name of the image
 image_name = 'rectangles.jpg'

# If the image is not in the current directory convert the 2nd page of treeleaf task pdf to jpg
if image_name not in os.listdir():
    print("Extracting image from treeleaf task pdf...")
    images = convert_from_path('treeleaf.pdf')
    images[1].save(image_name, 'JPEG')

# Read the image file
image = cv2.imread(image_name)

# Work with resized image as the original image is large
resized_image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)

# Using Canny's Edge detection algorithm to detect edges in the image
# Then Applying Simple Chain Approximation to find the contours and verifying if they are rectangles
# Then for each contour we look at the contours inside to find the line and their length
# Finally the lengths are sorted in descending order 
rectangles = find_rectangles_with_inside_line_length(resized_image)

# Draw rectangles and labels
for i, (x, y, w, h, line_length) in enumerate(rectangles):
#   cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cv2.putText(resized_image, str(i+1), (x + 10, y + h+22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Numbered Image", resized_image)
cv2.imwrite("output/numbering.png",resized_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
    

