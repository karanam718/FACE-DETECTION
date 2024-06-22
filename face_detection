import cv2
from google.colab.patches import cv2_imshow # Import the alternative function

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('/content/face.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2_imshow(img) # Pass only the image to cv2_imshow
# cv2.waitKey(0)  # Not needed in Colab
# cv2.destroyAllWindows() # Not needed in Colab
