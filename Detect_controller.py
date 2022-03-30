import threading
import numpy
from Detector import *

videos = ['4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4', '4ktestvid_Trim10sec.mp4']
line_positions = [[215, 215, 215, 215], [0, 0, 0, 0], [250, 250, 250, 250], [150, 150, 150, 150], [0, 0, 0, 0],
                  [250, 250, 250, 250]]
P = len(videos)

results = [[None] * 2] * P

threads = numpy.empty(P, dtype=object)

if __name__ == '__main__':
    detectors = Detector(videos, line_positions, results)

    for id in range(P):
        threads[id] = threading.Thread(target=detectors.vehicle_detection, args=(id,))
        threads[id].start()

    for id in range(P):
        threads[id].join()

    output = detectors.output()
    print(output, "Here")
cv2.destroyAllWindows()
