import csv
import datetime
from os.path import exists


def write_to_csv(output):
    header = ['Cam1_in', 'Cam1_out', 'Cam2_in', 'Cam2_out', 'Cam3_in', 'Cam3_out', 'Cam4_in', 'Cam4_out', 'Time']
    #data = [output[0][0], output[0][1], output[1][0], output[1][1], output[2][0], output[2][1], output[3][0],
            #output[3][1], datetime.datetime.now()]
    data = [output[0][0], output[0][1], datetime.datetime.now()]

    file_exists = exists('detector_db.csv')
    if file_exists:
        with open('detector_db.csv', 'a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(data)
            f.close()
    else:
        with open('detector_db.csv', 'w', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(header)
            writer_object.writerow(data)
            f.close()
