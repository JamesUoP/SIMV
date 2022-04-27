import threading
import pandas
import numpy
import cv2

from time import sleep
from cvzone.ClassificationModule import Classifier

# GOLBAL VARIABLES
feeds = ['Videos/4KTestVid_01.mp4']     # list of video/camera feeds
P = len(feeds)                          # number of feeds


# ---------------------------------------------- [ VEHICLE  DETECTION ] ---------------------------------------------- #
class Detection:

    def __init__(self):
        self.width_min = 40                     # minimum rectangle length
        self.height_min = 40                    # minimum rectangle length
        self.offset = 4                         # detection line offset
        self.line_cords = [0, 250, 215]         # line cords [x1,x2,y]
        self.fps = 100                          # frame rate

        self.classify = Classification()        # call to classification class

    def main(self, id):
        kernel = None
        detect = []
        vehicles = 0
        previous_frame = 0
        previous_x = 0
        backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        classList = [0] * len(list(open("MyModel/labels.txt")))

        cap = cv2.VideoCapture(feeds[id])

        while True:
            ret, frame = cap.read()
            tempo = float(1 / self.fps)
            sleep(tempo)
            if not ret:
                break

            # Apply the background object on the frame to get the segement mask
            fgmask = backgroundObject.apply(frame)

            _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, kernel, iterations=2)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frameCopy = frame.copy()

            for cnt in contours:

                if cv2.contourArea(cnt) > 400:
                    x, y, width, height = cv2.boundingRect(cnt)

                    cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    cv2.putText(frameCopy, 'Car Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.line(frame, (self.line_cords[0], self.line_cords[2]),
                     (self.line_cords[1], self.line_cords[2]), (255, 127, 0), 1)
            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_outline = (w >= self.width_min) and (h >= self.height_min)
                if not validate_outline:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = self.work_centre(x, y, w, h)
                detect.append(center)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

                for (x, y) in detect:
                    if vehicles == 0:
                        previous_x = x
                    if (((self.line_cords[2] + self.offset) > y > (self.line_cords[2] - self.offset)) and
                         (self.line_cords[0] < x < self.line_cords[1])):
                        previous_frame, previous_x, detect, vehicles = self.detection(id, cap, detect, vehicles, frame,
                                                                                      x, y, w, h, previous_frame,
                                                                                      previous_x, classList)
                        #print(str(detect))

                    if len(detect) == 10:  # when detect is of size 10 we clear for storage reasons
                        detect.clear()

            foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
            stacked = numpy.hstack((frame, foregroundPart, frameCopy))
            cv2.imshow('Detector Windows: {}'.format(id), stacked)

            k = cv2.waitKey(1) & 0xff

            if k == ord('q'):
                break
        #print("1")

        cap.release()

    def work_centre(self, x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    def screenshot(self, id, frame, x, y, w, h, vehicle_count, classList):

        # setting the sizes around the detection box
        sizes = [int(y - h / 2), int(y + h / 2), int(x - w / 2), int(x + w / 2)]
        img_height = sizes[1] - sizes[0]
        img_width = sizes[3] - sizes[2]

        cropped_image = frame[sizes[0]:sizes[1], sizes[2]:sizes[3]]

        # ensuring that the images are large enough to allow for easy checking
        if img_height < 150:
            img_height = 150
        if img_width < 150:
            img_width = 150

        cropped_image = cv2.resize(cropped_image, (img_height, img_width))
        # cv2.imshow("cropped image {}.{}".format(id, vehicle_count), cropped_image)

        index = self.classify.classifyAnImage(id, cropped_image)
        classList[index] += 1

    def detection(self, id, cap, detect, vehicle_count, frame, x, y, w, h, previous_frame, previous_x, classList):
        prev_frame = cap.get(1)
        prev_x = x

        if cap.get(1) > previous_frame + 20 or (x > previous_x + 50 or x < previous_x - 50):
            vehicle_count += 1
            cv2.line(frame, (self.line_cords[0], self.line_cords[2]),
                     (self.line_cords[1], self.line_cords[2]), (0, 127, 255), 1)
            #print("car is detected : " + str(vehicle_count) + " " + str(cap.get(1)))
            self.screenshot(id, frame, x, y, w, h, vehicle_count, classList)

        detect.remove((x, y))
        return prev_frame, prev_x, detect, vehicle_count


# -------------------------------------------- [ VEHICLE CLASSIFICATION ] -------------------------------------------- #
class Classification:

    def __init__(self):
        self.myClassifier = Classifier("MyModel/Keras_model.h5", "MyModel/labels.txt")
        self.vehicleNum = [0] * P

    def classifyAnImage(self, id, new_image):
        predictions, index = self.myClassifier.getPrediction(new_image)
        print("Prediction: {}   Classified as: {}".format(predictions, index))
        cv2.imshow("image {}.{}".format(id, self.vehicleNum[id]), new_image)
        self.vehicleNum[id] += 1
        return index


# --------------------------------------------------- [ MAIN RUN ] --------------------------------------------------- #
threads = numpy.empty(P, dtype=object)  # list of P number of threads

if __name__ == '__main__':
    detect = Detection()
    for id in range(P):
        threads[id] = threading.Thread(target=detect.main, args=(id,))
        threads[id].start()

    cv2.destroyAllWindows()