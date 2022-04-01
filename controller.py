import threading
import csv_writer
import os.path
import time
import numpy
from detector import *

videos = ['4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4']
line_positions = [[215, 215, 215, 215], [0, 0, 0, 0], [250, 250, 250, 250], [150, 150, 150, 150], [0, 0, 0, 0],
                  [250, 250, 250, 250]]
P = len(videos)
j = 0
repeat = 2
results = [[None] * 2] * P
threads = numpy.empty(P, dtype=object)

if __name__ == '__main__':

    while j < repeat:
        j += 1
        detectors = Detector(videos, line_positions, results)  # instantiate detector
        for id in range(P):
            threads[id] = threading.Thread(target=detectors.vehicle_detection, args=(id,))  # create threads
            threads[id].start()  # start threads

        for id in range(P):
            threads[id].join()  # join threads

        # output is then retrieved from all threads
        output = detectors.output()
        # output is formatted and written to csv
        csv_writer.write_to_csv(output)

        print(output)  # prints output

        cv2.destroyAllWindows()
