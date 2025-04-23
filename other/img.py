import cv2
import numpy as np

def create_aruco_corner_image(output_path="assets/aruco_corners.png", image_size=(800, 600), 
                              marker_size=100, marker_ids=[0, 1, 2, 3], margin=50):
    """
    Create an image with ArUco markers in each corner with white margins
    
    Parameters:
    -----------
    output_path : str
        Path to save the output image
    image_size : tuple
        Size of the output image (width, height)
    marker_size : int
        Size of each ArUco marker in pixels
    marker_ids : list
        IDs of the markers to use for each corner [top-left, top-right, bottom-right, bottom-left]
    margin : int
        Size of the white margin around the edge of the image in pixels
    """
    # Create a white image
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Define the corner positions with margins
    corners = [
        (margin, margin),                                          # Top-left
        (image_size[0] - marker_size - margin, margin),                # Top-right
        (image_size[0] - marker_size - margin, image_size[1] - marker_size - margin),  # Bottom-right
        (margin, image_size[1] - marker_size - margin)                 # Bottom-left
    ]
    
    # Draw each marker in its corner
    for i, (x, y) in enumerate(corners):
        marker_id = marker_ids[i]
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Convert marker to 3 channels if needed
        if len(marker_img.shape) == 2:
            marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Place the marker in the corner
        img[y:y+marker_size, x:x+marker_size] = marker_img
    
    # Add text to indicate marker IDs
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_positions = [
        (margin + marker_size + 10, margin + marker_size // 2),            # Top-left
        (image_size[0] - margin - marker_size - 70, margin + marker_size // 2),  # Top-right
        (image_size[0] - margin - marker_size - 70, image_size[1] - margin - marker_size // 2),  # Bottom-right
        (margin + marker_size + 10, image_size[1] - margin - marker_size // 2)   # Bottom-left
    ]
    
    for i, pos in enumerate(text_positions):
        cv2.putText(img, f"ID: {marker_ids[i]}", pos, font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"ArUco marker image saved to {output_path}")
    return img

if __name__ == "__main__":
    # Create the image and save it
    img = create_aruco_corner_image()
    
    # Display the image
    cv2.imshow("ArUco Markers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()