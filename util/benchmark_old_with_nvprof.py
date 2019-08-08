import os
import subprocess

benchmark_directory = "../equals"
original_file = "../equals/equals.cu"

grid_options = [10,16,50,75,100]
block_options = [32,64,96,128,160,192,224,256]
repetitions = 5
execution_count = len(grid_options) * len(block_options) * repetitions
grid_string = "#define BLOCK_COUNT "
block_string = "#define THREAD_COUNT "
kernel_name = "equalsKernel"
line_with_kernel_time_start = "test kernel: "
line_with_kernel_time_end = " ms"
log_occupancy_keyword = "Achieved Occupancy"
log_efficiency_keyword = "Warp Execution Efficiency"
log_error_output = "Number of matches is: 0"

os.chdir(benchmark_directory)

os.system("cp " + original_file + " ./base.cu")

output = ""
current_repetition = 1

for gi in range(0,len(grid_options)):
    for bi in range(0,len(block_options)):
        filename = "%d_%d" % (grid_options[gi], block_options[bi])
        os.system("sed \" s/%s/%s%d/g\" ./base.cu | sed \"s/%s/%s%d/g\" > %s.cu" % (grid_string, grid_string, grid_options[gi], block_string, block_string, block_options[bi], filename))
        os.system("nvcc %s.cu -o %s" % (filename, filename))
        output += "Grid size: %s, block size: %s\n" % (grid_options[gi], block_options[bi])

        kernel_times = []
        occupancies = []
        efficiencies = []
        successful = 1

        for repeated_tests in range(0,repetitions):
            execution_output = subprocess.check_output("nvprof --kernels ::%s: --metrics achieved_occupancy,warp_execution_efficiency --log-file temp_log ./%s" % (kernel_name, filename), shell=True)
            execution_output = execution_output.decode("utf-8")
            execution_output = execution_output.split("\n")

            if (execution_output[0] == log_error_output):
                print("Finished repetition " + str(current_repetition) + "/" + str(execution_count))
                current_repetition = current_repetition + 5

                successful = 0
                break

            time_line = list(filter(lambda x: x.startswith(line_with_kernel_time_start), execution_output))[0]
            time_string = time_line.replace(line_with_kernel_time_start, "").replace(line_with_kernel_time_end, "")
            time = float(time_string)
            kernel_times.append(time)

            with open("temp_log", "r") as logF:
                log_output = logF.read()

            log_output = log_output[log_output.find(log_occupancy_keyword):]
            log_output = log_output.replace(log_occupancy_keyword, "")
                
            occupancy = float(log_output.split()[2])
            occupancies.append(occupancy)

            log_output = log_output[log_output.find(log_efficiency_keyword):]
            log_output = log_output.replace(log_efficiency_keyword, "")

            warp_execution_efficiency = float(log_output.split()[2].strip("%"))/100
            efficiencies.append(warp_execution_efficiency)

            print("Finished repetition " + str(current_repetition) + "/" + str(execution_count))
            current_repetition = current_repetition + 1

        if (successful):
            output += "Kernel times: "
            output += str(kernel_times)
            output += "\nWarp execution efficiencies: "
            output += str(efficiencies)
            output += "\nAchieved occupancies: "
            output += str(occupancies)
            output += "\n"
            output += execution_output[0]
            output += "\nKernel: "
            output += "{0:.2f}".format(sum(kernel_times)/len(kernel_times))
            output += "\nWE-Efficiency: "
            output += "{0:.4f}".format(sum(efficiencies)/len(efficiencies))
            output += "\nOccupancy: "
            output += "{0:.4f}".format(sum(occupancies)/len(occupancies))
        else:
            output += "This configuration did not work!"

        output += "\n\n"

with open("./report.txt", 'w') as outF:
    outF.write(output)