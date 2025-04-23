import cv2
import numpy as np
import os

def generate_aruco_corner_image(output_path="assets/markers/aruco_corners.png", image_size=(800, 600), 
                              marker_size=100, marker_ids=[0, 1, 2, 3], margin=50):
    """
    Generate an image with ArUco markers in each corner with white margins and save it to assets/markers
    
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
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
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

def generate_individual_aruco_markers(output_dir="assets/markers", 
                                     marker_ids=[0, 1, 2, 3], 
                                     marker_size=200, 
                                     margin=50):
    """
    Generate individual ArUco marker images and save them separately
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the output images
    marker_ids : list
        List of marker IDs to generate
    marker_size : int
        Size of each ArUco marker in pixels
    margin : int
        Size of the white margin around the edge of each marker image in pixels
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    for marker_id in marker_ids:
        # Calculate total image size including margins
        total_size = marker_size + (2 * margin)
        
        # Create white image with margins
        img = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255
        
        # Generate marker
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Convert marker to 3 channels if needed
        if len(marker_img.shape) == 2:
            marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Place marker in the center of the image
        img[margin:margin+marker_size, margin:margin+marker_size] = marker_img
        
        # Add text to indicate marker ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (0, total_size)
        cv2.putText(img, f"ID: {marker_id}", text_position, font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Save the image
        output_path = os.path.join(output_dir, f"aruco_id_{marker_id}.png")
        cv2.imwrite(output_path, img)
        print(f"ArUco marker ID {marker_id} saved to {output_path}")
    
    print(f"Generated {len(marker_ids)} individual ArUco markers")

if __name__ == "__main__":
    # Generate the combined image with markers in corners
    generate_aruco_corner_image()
    
    # Generate individual marker images
    generate_individual_aruco_markers()
    
    print("All ArUco marker images have been generated successfully.")