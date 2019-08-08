import os
import subprocess

benchmark_directory = "../benchmark"
original_file = "../equals/equals.cu"
load_file = "../equals/load_db.cu"

selectivities = ["0.0025", "0.005", "0.01", "0.02", "0.04", "0.08", "0.16", "0.32", "0.64"]
grid_options = ["1000","2000","3000","4000","6000","8000","10000","20000","50000","100000","150000","200000"]
block_options = ["32","64","96","128","160","192","224","256","384","512","640","768"]
repetitions = 3
execution_count = len(selectivities) * len(grid_options) * len(block_options) * repetitions
block_string = "#define BLOCK_SIZE "
data_file_prefix = "../data/dblp/dblp_"
data_file_postfix = "_data.csv"
filename_string = "#define FILENAME "

os.chdir(benchmark_directory)
os.system("cp " + original_file + " ./base.cu")
os.system("cp " + load_file + " ./load_base.cu")

output = ""
current_repetition = 1
for si in range(0,len(selectivities)):
    load_filename = "load_temp"
    os.system("sed \'s:%s:%s\"%s%s%s\":g\' ./load_base.cu > %s.cu" % (filename_string, filename_string, data_file_prefix, selectivities[si], data_file_postfix, load_filename))
    os.system("nvcc %s.cu -o %s" % (load_filename, load_filename))
    os.system("./%s" % (load_filename))
    for bi in range(0,len(block_options)):
        filename = "temp"
        os.system("sed \"s/%s/%s%s/g\" ./base.cu > %s.cu" % (block_string, block_string, block_options[bi], filename))
        os.system("nvcc %s.cu -o %s" % (filename, filename))
        for gi in range(0,len(grid_options)):
            output += "G:%sB:%s\n" % (grid_options[gi], block_options[bi])

            number_of_trys = 1

            while True:
                kernel_times = []

                for repeated_tests in range(0,repetitions):
                    execution_output = subprocess.check_output("./%s %s" % (filename, grid_options[gi]), shell=True)
                    execution_output = execution_output.decode("utf-8")
                    time = float(execution_output)
                    kernel_times.append(time)

                    print("Finished repetition " + str(current_repetition) + "/" + str(execution_count))
                    current_repetition = current_repetition + 1
                
                average = sum(kernel_times)/len(kernel_times)

                if (max(kernel_times) - min(kernel_times)) / average < 0.1:
                    break
                
                if number_of_trys == 3:
                    output += "High fluctuation\n"
                    break
                
                number_of_trys = number_of_trys + 1

            output += "{0:.2f}\n".format(average)

with open("./report.txt", 'w') as outF:
    outF.write(output)
