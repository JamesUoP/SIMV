import csv
import datetime
from os.path import exists


def write_to_csv(output):
    header = ['Cam1_in', 'Cam1_out', 'Cam2_in', 'Cam2_out', 'Cam3_in', 'Cam3_out', 'Cam4_in', 'Cam4_out', 'Time'] # header for csv file
    #data = [output[0][0], output[0][1], output[1][0], output[1][1], output[2][0], output[2][1], output[3][0],
            #output[3][1], datetime.datetime.now()]
    data = [output[0][0], output[0][1], datetime.datetime.now()] # results data for cars going into the one video and going out then the time of data input

    file_exists = exists('detector_db.csv') # checks if csv file exits
    if file_exists: # if exits it will open the file and write the new data on a new line within the csv
        with open('detector_db.csv', 'a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(data)
            f.close()
    else: # if the csv does not exist it will create the file and then write to the csv file
        with open('detector_db.csv', 'w', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(header)
            writer_object.writerow(data)
            f.close()
