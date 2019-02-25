import csv
simulations = [1, 2]
steps = [0,1,2,3]
with open('training_data_test.csv', mode='w') as training_file:
    filewriter = csv.writer(training_file, delimiter=':', escapechar='"', quoting=csv.QUOTE_NONE)
    filewriter.writerow(['Action ', ' Observation'])
    for sim in simulations:
        filewriter.writerow(['Simulation ', sim])
        for step in steps:
            filewriter.writerow([[0,1], [1,2,3,4,5,6]])

