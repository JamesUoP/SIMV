import threading
import pandas
import numpy
import time
import cv2

from cvzone.ClassificationModule import Classifier

# ----------------------------------------------- [ GLOBAL VARIABLES ] ----------------------------------------------- #
feeds = ['Videos/4KTestVid_01.mp4',
         'Videos/4KTestVid_01.mp4']                             # LIST OF VIDEO/CAMERA FEEDS
P = len(feeds)                                                  # NUMBER OF FEEDS
VP = len(list(open('MyModel/labels.txt')))                      # NUMBER OF VEHICLE TYPES
test_mode = True                                                # ENABLE/DISABLE TEST MODE
cutoff_timer = 5                                                # CUTOFF TIMER FOR OPERATION CYCLE
df = pandas.read_csv('database.csv')                            # READS DATA FROM CSV FILE


# ---------------------------------------------- [ VEHICLE  DETECTION ] ---------------------------------------------- #
class Detection:

    def __init__(self):
        self.width_min = 40                                     # MINIMUM RECTANGLE WIDTH
        self.height_min = 40                                    # MINIMUM RECTANGLE HEIGHT
        self.offset = 4                                         # DETECTION LINE OFFSET
        self.fps = 100                                          # FRAME RATE
        self.synch = threading.Barrier(P)                       # BARRIER TO SYNCH THREADS
        self.classify = Classification()                        # INITIATES CLASS 'CLASSIFICATION'
        self.classList = numpy.zeros((P, (2 * VP)), dtype=int)  # MATRIX OF CLASSIFIED VEHICLES
                                                                # (0...VP-1 = IN | VP...2VP-1 = OUT)

    def main(self, id, lines):
        kernel = None
        detections = []                                         # LIST OF DETECTIONS
        vehicles = [0, 0]                                       # NUMBER OF VEHICLES DETECTED (IN,OUT)
        previous_x = [0, 0]                                     # PREVIOUS X VALUES (IN,OUT)
        previous_frame = [0, 0]                                 # PREVIOUS CAPTURE FRAME
        my_line_cords = lines                                   # DETECTION LINES FOR THREAD FEED
        cap = cv2.VideoCapture(feeds[id])                       # CAPTURE OF FEED
        backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        if id == 0:                                             # THREAD-0 RETURNS START TIME (HH:MM) & DATE (YYY/MM/DD)
            start_time, start_date = self.format_time()
        timer = time.time()                                     # SETS TIMER TO CURRENT TIME (SECS)

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

            # CALCULATE CONTOURS
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frameCopy = frame.copy()

            for cnt in contours:
                if cv2.contourArea(cnt) > 400:
                    x, y, width, height = cv2.boundingRect(cnt)

                    cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    cv2.putText(frameCopy, 'Car Detected', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.line(frameCopy, (my_line_cords[0], my_line_cords[2]),
                                (my_line_cords[1], my_line_cords[2]), (255, 0, 0), 1)     # IN DETECTION LINE
            cv2.line(frameCopy, (my_line_cords[3], my_line_cords[5]),
                                (my_line_cords[4], my_line_cords[5]), (0, 255, 0), 1)     # OUT DETECTION LINE

            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_outline = (w >= self.width_min) and (h >= self.height_min)
                if not validate_outline:
                    continue

                center = self.work_centre(x, y, w, h)       # CALL TO FUNCTION 'WORK CENTRE'
                detections.append(center)

#                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frameCopy, center, 4, (0, 0, 255), -1)

                for (x, y) in detections:
                    for i in range(2):          # 0 = IN | 1 = OUT
                        eqt = (3 * i)           # EQUATION FOR CALCULATING INDEX VALUE
                        if (((my_line_cords[2+eqt] + self.offset) > y >
                             (my_line_cords[2+eqt] - self.offset)) and (my_line_cords[0+eqt] < x <
                                                                        my_line_cords[1+eqt])):

                            if vehicles[i] == 0:
                                previous_x[i] = x

                            # CALL TO FUNCTION 'DETECTION'
                            detections, vehicles[i], previous_x[i], previous_frame[i] = self.detection(id, i, cap,
                                                                                                    detections,
                                                                                                    vehicles[i],
                                                                                                    previous_x[i],
                                                                                                    previous_frame[i],
                                                                                                    frame, x, y, w, h)

                if len(detections) == 10:       # CLEAR DETECT WHEN SIZE=10 TO MINIMISE STORAGE
                    detections.clear()

            if test_mode:
                foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
                stacked = numpy.hstack((frame, foregroundPart, frameCopy))
                cv2.imshow('Detector Windows: {}'.format(id), stacked)

            if (time.time() - timer) >= cutoff_timer:
                self.synch.wait()                               # SYNCH BEFORE CSV UPDATE
                if id == 0:                                     # THREAD-0 UPDATES THE CSV FILE
                    print(self.classList)
                    self.write_to_csv(start_time, start_date)
                self.synch.wait()                               # SYNCH AFTER CSV UPDATE

                for i in range(2*VP):                           # EACH THREAD CLEARS CLASSIFIED DETECTIONS
                    self.classList[id][i] = 0
                if id == 0:                                     # THREAD-0 RETURNS NEW START TIME & DATE
                    start_time, start_date = self.format_time()
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

    def write_to_csv(self, sTime, sDate):
        eTime,eDate = self.format_time()        # RETURNS END TIME & DATE

        for i in range(P):
            df.loc[len(df.index)] = [i, self.classList[i][0], self.classList[i][1],
                                        self.classList[i][2], self.classList[i][3], sTime, eTime, sDate, eDate]
            df.to_csv('database.csv', index=False)
        print(df)

    def screenshot(self, id, direction, vehicles, frame, x, y, w, h):
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
        index += (VP * direction)
        self.classList[id][index] += 1

    def detection(self, id, direction, cap, detections, vehicles, previous_x, previous_frame, frame, x, y, w, h):
        prev_frame = cap.get(1)
        prev_x = x

        if cap.get(1) > previous_frame + 20 or (x > previous_x + 50 or x < previous_x - 50):
            vehicles += 1

#            cv2.line(frame, (my_line_cords[0], my_line_cords[2]),
#                            (my_line_cords[1], my_line_cords[2]), (0, 127, 255), 1)         # IN DETECTION LINE
#            cv2.line(frame, (my_line_cords[3], my_line_cords[5]),
#                            (my_line_cords[4], my_line_cords[5]), (0, 127, 255), 1)         # OUT DETECTION LINE
#            print("car is detected : " + str(vehicles) + " " + str(cap.get(1)))

            self.screenshot(id, direction, vehicles, frame, x, y, w, h)        # CALL TO FUNCTION 'SCREENSHOT'

        #detections.remove((x, y))
        return detections, vehicles, prev_x, prev_frame

# -------------------------------------------- [ VEHICLE CLASSIFICATION ] -------------------------------------------- #
class Classification:

    def __init__(self):
        self.myClassifier = Classifier("MyModel/Keras_model.h5",
                                       "MyModel/labels.txt")        # CALL TO CVZONE CLASS 'CLASSIFIER'
        self.vehicleNum = [0] * P                                   # NUMBER OF CLASSIFIED VEHICLE

    def classifyAnImage(self, id, new_image):
        predictions, index = self.myClassifier.getPrediction(new_image)

        # DISPLAYS OUTPUTS FOR CLASSIFICATION
        print("Image: {}    Prediction: {}   Classified as: {}".format(self.vehicleNum[id], predictions, index))
        if test_mode:
            cv2.imshow("image {}.{}".format(id, self.vehicleNum[id]), new_image)

        self.vehicleNum[id] += 1
        return index


# --------------------------------------------------- [ MAIN RUN ] --------------------------------------------------- #
line_cords = [[0, 250, 215, 260, 365, 215],
              [0, 250, 215, 260, 365, 215]]                 # CO-ORDS FOR THE DETECTION LINES

# CHECKS IF THE MATRIX EQUALS: Px6
if len(line_cords) == P:
    for row in line_cords:
        columns = len(row)
        if columns != 6:
            raise ValueError('Number of values in row is not 6')
else:
    raise ValueError('Number of rows is not equal to {}'.format(P))

threads = numpy.empty(P, dtype=object)                      # LIST OF P NUMBER OF THREADS

if __name__ == '__main__':
    detect = Detection()                                    # INITIATES CLASS 'DETECTION'
    for i in range(P):
        threads[i] = threading.Thread(target=detect.main, args=(i, line_cords[i],))
        threads[i].start()

    for i in range(P):
        threads[i].join()

    cv2.destroyAllWindows()