import threading
import pandas
import numpy
import time
import cv2

from cvzone.ClassificationModule import Classifier

# ----------------------------------------------- [ GLOBAL VARIABLES ] ----------------------------------------------- #
feeds = ['Videos/4KTestVid_01.mp4',
         'Videos/4KTestVid_01.mp4']                         # LIST OF VIDEO/CAMERA FEEDS
P = len(feeds)                                              # NUMBER OF FEEDS
VP = len(list(open('MyModel/labels.txt')))                  # NUMBER OF VEHICLE TYPES

cutoff_timer = 5                                            # CUTOFF TIMER FOR OPERATION CYCLE
df = pandas.read_csv('database.csv').set_index(' ')         # READS DATA FROM CSV FILE


# ---------------------------------------------- [ VEHICLE  DETECTION ] ---------------------------------------------- #
class Detection:

    def __init__(self):
        self.width_min = 40                                 # MINIMUM RECTANGLE WIDTH
        self.height_min = 40                                # MINIMUM RECTANGLE HEIGHT
        self.offset = 4                                     # DETECTION LINE OFFSET
        self.line_cords = [0, 250, 215]                     # LINE CO-ORDS [x1,x2,y]
        self.fps = 100                                      # FRAME RATE
        self.synch = threading.Barrier(P)                   # BARRIER TO SYNCH THREADS

        self.classify = Classification()                    # INITIATES CLASS 'CLASSIFICATION'
        self.classList = numpy.zeros((P, VP), dtype=int)    # MATRIX OF CLASSIFIED VEHICLES

    def main(self, id):
        kernel = None
        detect = []                                         # LIST OF DETECTIONS
        vehicles = 0                                        # NUMBER OF VEHICLES DETECTED
        previous_x = 0                                      # PREVIOUS X VALUE
        previous_frame = 0                                  # PREVIOUS CAPTURE FRAME
        cap = cv2.VideoCapture(feeds[id])                   # CAPTURE OF FEED
        backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        if id == 0:
            start_time, start_date = self.format_time()     # RETURNS START TIME (HH:MM) & DATE (YYY/MM/DD)
        timer = time.time()                                 # SETS TIMER TO CURRENT TIME (SECS)

        print(df.tail(P))

        while True:
            ret, frame = cap.read()
            tempo = float(1 / self.fps)
            time.sleep(tempo)
            if not ret:
                break

            # APPLY BACKGROUND OBJECT ON FRAME TO ACQUIRE SEGMENT MASK
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
                    cv2.putText(frameCopy, 'Car Detected', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

#            cv2.line(frame, (self.line_cords[0], self.line_cords[2]),
#                            (self.line_cords[1], self.line_cords[2]), (255, 127, 0), 1)

            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_outline = (w >= self.width_min) and (h >= self.height_min)
                if not validate_outline:
                    continue

                center = self.work_centre(x, y, w, h)           # CALL TO FUNCTION 'WORK CENTRE'
                detect.append(center)

#                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                cv2.circle(frame, center, 4, (0, 0, 255), -1)

                for (x, y) in detect:
                    if vehicles == 0:
                        self.previous_x = x
                    if (((self.line_cords[2] + self.offset) > y >
                         (self.line_cords[2] - self.offset)) and (self.line_cords[0] < x < self.line_cords[1])):

                        # CALL TO FUNCTION 'DETECTION'
                        detect, vehicles, previous_x, previous_frame = self.detection(id, cap, detect, vehicles,
                                                                                      previous_x, previous_frame,
                                                                                      frame, x, y, w, h)
#                        print(str(detect))

                    if len(detect) == 10:                       # CLEAR DETECT WHEN SIZE=10 TO MINIMISE STORAGE
                        detect.clear()

            foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
            stacked = numpy.hstack((frame, foregroundPart, frameCopy))
            cv2.imshow('Detector Windows: {}'.format(id), stacked)

            if (time.time() - timer) >= cutoff_timer:
                self.synch.wait()                               # SYNCH BEFORE CSV UPDATE

                """ CREATE FOUR NEW ROWS IN DATAFRAME """

                if id == 0:                                     # THREAD-0 UPDATES THE CSV FILE
                    end_time, end_date = self.format_time()     # RETURNS END TIME & DATE
                    self.write_to_csv(start_time, start_date, end_time, end_date)
                self.synch.wait()                               # SYNCH AFTER CSV UPDATE

                for i in range(VP):                             # EACH THREAD CLEARS CLASSIFIED DETECTIONS
                    self.classList[id][i] = 0
                start_time, start_date = self.format_time()     # RETURNS NEW START TIME & DATE
                timer = time.time()                             # RESETS TIMER

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break

        cap.release()

    def format_time(self):
        newtime = time.localtime()
        myTime = ('{}:{}'.format(str(newtime[3]).zfill(2), str(newtime[4]).zfill(2)))
        myDate = ('{}/{}/{}'.format(newtime[0], str(newtime[1]).zfill(2), str(newtime[2]).zfill(2)))
        return myTime, myDate

    def work_centre(self, x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    def write_to_csv(self, sTime, sDate, eTime, eDate):
        for i in range(P):
            df.loc[len(df.index)] = [i, self.classList[i][0], self.classList[i][1], sTime, sDate, eTime, eDate]
            df.to_csv('database.csv', index=True)
        print(df)

    def screenshot(self, id, vehicles, frame, x, y, w, h):
        # SETS THE SIZE AROUND DETECTION BOX
        sizes = [int(y - h / 2), int(y + h / 2), int(x - w / 2), int(x + w / 2)]
        cropped_image = frame[sizes[0]:sizes[1], sizes[2]:sizes[3]]

        img_height = sizes[1] - sizes[0]
        img_width = sizes[3] - sizes[2]

        # SETS MINIMUM SIZE FOR CLEARER DISPLAY
        if img_height < 150:
            img_height = 150
        if img_width < 150:
            img_width = 150

        cropped_image = cv2.resize(cropped_image, (img_height, img_width))
#        cv2.imshow("cropped image {}.{}".format(id, vehicles), cropped_image)

        index = self.classify.classifyAnImage(id, cropped_image)        # CALL TO CLASS 'CLASSIFICATION'
        self.classList[id][index] += 1

    def detection(self, id, cap, detect, vehicles, previous_x, previous_frame, frame, x, y, w, h):
        prev_frame = cap.get(1)
        prev_x = x

        if cap.get(1) > previous_frame + 20 or (x > previous_x + 50 or x < previous_x - 50):
            vehicles += 1

#            cv2.line(frame, (self.line_cords[0], self.line_cords[2]),
#                            (self.line_cords[1], self.line_cords[2]), (0, 127, 255), 1)
#            print("car is detected : " + str(vehicles) + " " + str(cap.get(1)))

            self.screenshot(id, vehicles, frame, x, y, w, h)            # CALL TO FUNCTION 'SCREENSHOT'

        detect.remove((x, y))
        return detect, vehicles, prev_x, prev_frame

# -------------------------------------------- [ VEHICLE CLASSIFICATION ] -------------------------------------------- #
class Classification:

    def __init__(self):
        self.myClassifier = Classifier("MyModel/Keras_model.h5",
                                       "MyModel/labels.txt")    # CALL TO CVZONE CLASS 'CLASSIFIER'
        self.vehicleNum = [0] * P                               # NUMBER OF CLASSIFIED VEHICLE

    def classifyAnImage(self, id, new_image):
        predictions, index = self.myClassifier.getPrediction(new_image)

        # DISPLAYS OUTPUTS FOR CLASSIFICATION
        print("Image: {}    Prediction: {}   Classified as: {}".format(self.vehicleNum[id], predictions, index))
        cv2.imshow("image {}.{}".format(id, self.vehicleNum[id]), new_image)

        self.vehicleNum[id] += 1
        return index


# --------------------------------------------------- [ MAIN RUN ] --------------------------------------------------- #
threads = numpy.empty(P, dtype=object)                      # LIST OF P NUMBER OF THREADS

if __name__ == '__main__':
    detect = Detection()                                    # INITIATES CLASS 'DETECTION'
    for i in range(P):
        threads[i] = threading.Thread(target=detect.main, args=(i,))
        threads[i].start()

    for i in range(P):
        threads[i].join()

    cv2.destroyAllWindows()
