import cv2
import numpy as np

def asciiConvert(frame, scale):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1] // scale
        height = gray.shape[0] // scale
        resized = cv2.resize(gray, (width, height))
        ascii_chars = "@@@***___ "  # Caracteres do mais escuro para o mais claro
        ascii_image = []

        for y in range(resized.shape[0]):
            line = ""
            for x in range(resized.shape[1]):
                pixel_value = resized[y, x]
                line += ascii_chars[pixel_value // 32]  # Escala dos valores de pixels
            ascii_image.append(line)

        return ascii_image
    except Exception as e:
        print(f"Error in asciiConvert: {e}")
        return []

def asciiArtToImage(ascii_image, scale):
    try:
        height = len(ascii_image)
        width = len(ascii_image[0])
        img = np.ones((height * scale, width * scale, 3), dtype=np.uint8) * 255

        for y, line in enumerate(ascii_image):
            for x, char in enumerate(line):
                if char == '@':
                    cv2.putText(img, char, (x * scale + scale // 4, (y + 1) * scale - scale // 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return img
    except Exception as e:
        print(f"Error in asciiArtToImage: {e}")
        return np.zeros((1, 1, 3), dtype=np.uint8)

def drawBoundingBoxes(img, ascii_image, scale):
    try:
        height = len(ascii_image)
        width = len(ascii_image[0])
        binary_img = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                if ascii_image[y][x] == '@':
                    binary_img[y, x] = 255

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_distance = float('inf')
        best_contour = None
        best_rect = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            top_left = (x * scale, y * scale)
            bottom_right = ((x + w) * scale, (y + h) * scale)
            distance = img.shape[0] - bottom_right[1]
            
            if distance < min_distance:
                min_distance = distance
                best_contour = (top_left, bottom_right)
                best_rect = (x, y, w, h)

        if best_contour:
            top_left, bottom_right = best_contour
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

            if best_rect:
                x, y, w, h = best_rect
                square_size = min(w, h) // 5  # Ajuste a quantidade de quadrados
                for i in range(0, w, square_size):
                    for j in range(0, h, square_size):
                        sx, sy = x + i, y + j
                        ex, ey = sx + square_size, sy + square_size
                        cv2.rectangle(img, (sx * scale, sy * scale), (ex * scale, ey * scale), (0, 255, 0), 1)

        return img, min_distance
    except Exception as e:
        print(f"Error in drawBoundingBoxes: {e}")
        return img, float('inf')

def drawVerticalLineAndDistance(img, ascii_image, scale):
    try:
        height = len(ascii_image)
        width = len(ascii_image[0])
        
        middle_x = width * scale // 2
        cv2.line(img, (middle_x, 0), (middle_x, height * scale), (0, 0, 255), 1)
        
        max_distance = 300
        bottom_y = height
        distances = []

        for y, line in enumerate(ascii_image):
            for x, char in enumerate(line):
                if char == '@':
                    pixel_y = y * scale
                    distance = bottom_y * scale - pixel_y
                    if distance <= max_distance:
                        distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            cv2.putText(img, f"Avg Distance: {int(avg_distance)} px", (10, height * scale - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        return img
    except Exception as e:
        print(f"Error in drawVerticalLineAndDistance: {e}")
        return img

def main():
    scale = 4

    try:
        webCam = cv2.VideoCapture(1)
        if not webCam.isOpened():
            raise Exception("Could not open webcam.")

        while True:
            ret, frame = webCam.read()
            if not ret:
                print("Failed to grab frame.")
                break

            ascii_image = asciiConvert(frame, scale)
            ascii_art_img = asciiArtToImage(ascii_image, scale)
            ascii_art_img, min_distance = drawBoundingBoxes(ascii_art_img, ascii_image, scale)
            ascii_art_img = drawVerticalLineAndDistance(ascii_art_img, ascii_image, scale)

            cv2.imshow('ASCII Art Feed', ascii_art_img)

            if min_distance != float('inf'):
                cv2.putText(ascii_art_img, f"Closest Rect Distance: {int(min_distance)} px", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in main: {e}")
    
    finally:
        webCam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
