import argparse
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

def analyze_emotions(image_path):
    # Perform emotion analysis using DeepFace
    results = DeepFace.analyze(image_path, actions=['emotion'])
    
    if isinstance(results, list):
        results = results[0]  # Use the first detected face
    
    # Print the detected emotions
    print("Emotions detected:")
    for emotion, score in results['emotion'].items():
        print(f"{emotion}: {score}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return
    
    # Get the bounding box and dominant emotion
    bounding_box = results['region']
    dominant_emotion = max(results['emotion'], key=results['emotion'].get)
    
    # Draw a rectangle around the detected face
    x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['w'], bounding_box['h']
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text with the dominant emotion
    cv2.putText(image, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image with the bounding box and emotion label
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze emotions in an image using DeepFace")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Analyze emotions in the provided image
    analyze_emotions(args.image_path)

if __name__ == "__main__":
    main()
