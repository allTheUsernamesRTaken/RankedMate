import cv2
import numpy as np
import sys
import os

def detect_square_and_sum_brightness(image_path):
    """
    Detects a black square on white paper and sums the brightness of pixels within it.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        tuple: (total_brightness, bounding_box_coords, processed_image)
    """
    # Read the image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None, None, None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None, None, None
    
    # Create a copy for processing
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    # This helps separate the black square from the white paper background
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Alternative: Use Otsu's thresholding if adaptive doesn't work well
    # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in the image.")
        return None, None, original
    
    # Find the largest contour (should be the black square)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    # This helps handle slight distortions
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Draw the detected contour on the original image
    result_img = original.copy()
    cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Create a mask for the detected square region
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)
    
    # Extract the region of interest (ROI) from the original grayscale image
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Sum the brightness values within the detected square
    # Only count pixels that are actually in the square (non-zero in mask)
    total_brightness = np.sum(roi[mask > 0])
    
    # Count the number of pixels in the square for average calculation
    pixel_count = np.count_nonzero(mask)
    average_brightness = total_brightness / pixel_count if pixel_count > 0 else 0
    
    print(f"\nSquare Detection Results:")
    print(f"  Total brightness: {total_brightness}")
    print(f"  Average brightness: {average_brightness:.2f}")
    print(f"  Pixel count: {pixel_count}")
    print(f"  Bounding box: x={x}, y={y}, width={w}, height={h}")
    print(f"  Contour vertices: {len(approx)} points")
    
    return total_brightness, (x, y, w, h), result_img


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_square.py <image_path>")
        print("\nExample: python detect_square.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    total_brightness, bbox, result_img = detect_square_and_sum_brightness(image_path)
    
    if total_brightness is not None:
        # Save the result image with detected square
        output_path = "detected_square_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"\nResult image saved to: {output_path}")
        
        # Display the result (optional - may not work in all environments)
        try:
            cv2.imshow("Detected Square", result_img)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("(Display window not available - result saved to file)")


if __name__ == "__main__":
    main()

