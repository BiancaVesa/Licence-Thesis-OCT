import glob
import pandas as pd
import pickle
import xml.etree.ElementTree as ET


def generate_oct_info_dataframe():
    """
    Parses the folder that has all the OCTs and stores all the information in a dataframe.
    Writes dataframe object in a file.
    """

    patient_paths = glob.glob("D:/AN4/Licenta/Datasets/AMD/UDF Volume/*")
    image_paths = []
    image_indexes = []
    patient_numbers = []
    patient_ids = []
    visit_numbers = []
    visit_dates = []
    eye_encodings = []
    data = []

    for p in patient_paths:
        visits = glob.glob(p + "/*")
        patient = p.rsplit('Pacient ', 1)[1]
        for v in visits:
            eyes = glob.glob(v + "/*")
            visit_nr = v.rsplit('Vizita ', 1)[1].rsplit(' - ', 1)[0]
            visit_date = v.rsplit(' - ', 1)[1]
            for e in eyes:
                xmls = glob.glob(e + "/*.xml")
                images = glob.glob(e + "/*.tif")
                eye = e.rsplit('\\', 1)[1]
                for x in xmls:
                    tree = ET.parse(x)
                    root = tree.getroot()
                    foundID = False
                    fundus_image = True
                    for child in root.iter():
                        # stocam id-ul pacientului pt acuitate
                        if child.tag in 'ID' and foundID == False:
                            patientID = child.text
                            foundID = True
                        if child.tag in 'ExamURL':
                            # nu pastram fundus image
                            if fundus_image == False:
                                # in xml imaginile sunt in ordine cronologica
                                image_name = child.text.rsplit('\\', 1)[1]
                                ki = 0
                                step = 1
                                fast_bscan_size = 26
                                bscan_size = len(images)
                                if bscan_size != fast_bscan_size:
                                    step = 2
                                for i in range(0, bscan_size, step):
                                    # alegem doar 25 de imagini
                                    if ki < fast_bscan_size:
                                        if image_name in images[i]:
                                            image_paths.append(image_name)
                                            image_indexes.append(ki)
                                            patient_numbers.append(patient)
                                            patient_ids.append(patientID)
                                            visit_numbers.append(visit_nr)
                                            visit_dates.append(visit_date)
                                            if 'OD' in eye:
                                                eye_encodings.append('right')
                                            else:
                                                eye_encodings.append('left')
                                        ki += 1
                                if bscan_size == 50:
                                    if image_name in images[49]:
                                        image_paths.append(image_name)
                                        image_indexes.append(25)
                                        patient_numbers.append(patient)
                                        patient_ids.append(patientID)
                                        visit_numbers.append(visit_nr)
                                        visit_dates.append(visit_date)
                                        if 'OD' in eye:
                                            eye_encodings.append('right')
                                        else:
                                            eye_encodings.append('left')

                            fundus_image = False

    k = 0
    for i in image_paths:
        data.append([image_paths[k], image_indexes[k], patient_numbers[k], patient_ids[k], visit_numbers[k], visit_dates[k], eye_encodings[k]])
        k +=  1
    df = pd.DataFrame(data, columns = ['Image name', 'Image index','Patient folder', 'Patient ID', 'Visit', 'Date', 'Eye'])

    with open('dataframe.txt', 'wb') as dataframe_file:
        pickle.dump(df, dataframe_file)


def convert_acuity(x):
    """
    Function to convert acuity values in decimal form
    """
    acuity_snellen = ['p.l.', '20/2000', '20/800', '20/400', '20/200', '20/160', '20/125', '20/100', '20/80', '20/63', '20/50',
                      '20/40', '20/32', '20/25', '20/20']
    acuity_decimal = [0.0, 0.01, 0.025, 0.05, 0.1, 0.125, 0.16, 0.2, 0.25, 0.32, 0.4, 0.5, 0.63, 0.8, 1]

    y = x
    if '-' in x:
        x = x.split('-')[0]
    if x == 'N/A':
        x = '-1.0'
    if x == '1':
        x = '1.0'
    if '/' in x or x == 'p.l.':
        return acuity_decimal[acuity_snellen.index(x)]
    else:
        if '.' in x:
            return float(x)
        else:
            return 0.0

def generate_acuity_dataframe():
    """
    Parses the csv file that has all the acuity values and stores all the information in a dataframe.
    Writes dataframe object in a file.
    """
    nr_rows = 94 * 3 # nr de linii care contin informatie in fisier
    k_row = 0
    first_row = False # pe prima linie nu sunt informatii utile

    patient_ids = []
    patient_folders = []
    visit_dates = []
    eyes = []
    acuities = []
    acuities_types = []

    with open('D:/AN4/Licenta/Datasets/AMD/Acuity/DMLVAVcuID.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        acuity_dates = []
        patient_id = 0
        patient_folder = 0
        for row in csv_reader:
            if first_row:
                if k_row < nr_rows:
                    k_item = 0
                    for item in row:
                        if k_row % 3 == 0:
                            if k_item == 0:
                                patient_folder = item
                            if k_item == 1:
                                patient_id = item
                            if item != '':
                                acuity_dates.append(item)
                        if k_row % 3 == 1:
                            if k_item > 1 and item != '':
                                patient_folders.append(patient_folder)
                                patient_ids.append(patient_id)
                                visit_dates.append(acuity_dates[k_item])
                                eyes.append('OD')
                                acuities.append(item.split()[0])
                                if len(item.split()) > 1:
                                    acuities_types.append(item.split()[1])
                                else:
                                    acuities_types.append('')
                        if k_row % 3 == 2 :
                            if k_item > 1 and item != '':
                                patient_folders.append(patient_folder)
                                patient_ids.append(patient_id)
                                eyes.append('OS')
                                visit_dates.append(acuity_dates[k_item])
                                acuities.append(item.split()[0])
                                if len(item.split()) > 1:
                                    acuities_types.append(item.split()[1])
                                else:
                                    acuities_types.append('')
                            if k_item > 1 and item == '':
                                acuity_dates = []
                        k_item += 1
                else:
                    break
                k_row += 1
            first_row = True

    data_acuity = []
    for k in range(0, len(patient_ids)):
        data_acuity.append([patient_folders[k], patient_ids[k], visit_dates[k], eyes[k], acuities[k], acuities_types[k]])

    acuity_df = pd.DataFrame(data_acuity, columns = ['Patient folder', 'Patient ID', 'Visit Date', 'Eye', 'Acuity', 'Acuity types'])
    acuity_df['Acuity'] = acuity_df['Acuity'].map(lambda x: convert_acuity(x))

    with open('acuity.txt', 'wb') as dataframe_file:
        pickle.dump(acuity_df, dataframe_file)
