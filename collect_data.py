import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(1, number_of_classes):

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Sedang mengambil data untuk kelas {}...'.format(j))

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Determine the size of the square (1:1 aspect ratio)
        min_dim = min(width, height)

        # Calculate the coordinates to crop the center
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        cropped_frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

        cv2.imshow('frame', cropped_frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), cropped_frame)

        counter += 1
    print('Kelas {}'.format(j)+' selesai')

cap.release()
cv2.destroyAllWindows()