import threading
import csv_writer
import numpy
from detector import *


# videos = ['TestVideoShort.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4']
videos = ['Videos/TestVideoShort.mp4','Videos/TestVideoShort.mp4','Videos/TestVideoShort.mp4','Videos/TestVideoShort.mp4']
line_positions = [[215, 215, 215, 215], [0, 0, 0, 0], [250, 250, 250, 250], [150, 150, 150, 150], [0, 0, 0, 0],
                  [250, 250, 250, 250]]
P = len(videos)
j = 0
repeat = 4
results = [[None] * 2] * P # results in list dependant on number of input videos/feeds
threads = numpy.empty(P, dtype=object) # number of threads

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

        cv2.destroyAllWindows() # closes windows
