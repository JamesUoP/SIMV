import csv
import datetime
from os.path import exists


def write_to_csv(output):
    header = ['Cam1_in', 'Cam1_out', 'Cam2_in', 'Cam2_out', 'Cam3_in', 'Cam3_out', 'Cam4_in', 'Cam4_out'] #Header of csv file
    data = [output[0][0], output[0][1], output[1][0], output[1][1], output[2][0], output[2][1], output[3][0], 
            output[3][1], datetime.datetime.now()] #Data formatted for csv file

    file_exists = exists('detector_db.csv') #Checks if the file exists 
    if file_exists: #If file exists it will write to the file and add the current data
        with open('detector_db.csv', 'a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(data)
            f.close()
    else: #If file does not exist it will create the file and will add the current data
        with open('detector_db.csv', 'w', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(header)
            writer_object.writerow(data)
            f.close()
