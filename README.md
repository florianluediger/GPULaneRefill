# GPULaneRefill

_Benchmarking applications to analyze efficient string processing with database queries on highly parallel hardware._

This repository contains the source code I wrote as part of my [masters thesis](https://github.com/florianluediger/StringverarbeitungGrafikkarten). 
With this code, you can perform your own benchmarks for testing the performance of string processing operations on NVIDIA GPUs.
If you speak German, you can also have a look at the results I achieved in my [masters thesis](https://github.com/florianluediger/StringverarbeitungGrafikkarten).

Some parts of this repository are authored by Henning Funke.
These parts will be explicitly marked with an author notice.

If you have any questions, feel free to contact me via E-Mail.

## Perform a benchmark using the DBLP dataset as an example

### Set up the development environment

To perform these benchmarks, you need an NVIDIA GPU, Linux and the CUDA toolkit installed.
You should also clone this repository which provides you with the tools to generate large datasets and perform the benchmarks.

```bash
git clone https://github.com/florianluediger/GPULaneRefill.git
```

### Download the DBLP dataset and modify it to our needs

```bash
# enter the root directory of the repository
cd GPULaneRefill

# download the DBLP dataset from their website
wget -P data https://dblp.org/xml/dblp.xml.gz

# extract the DBLP dataset
gunzip data/dblp.xml.gz

# extract the titles from the dataset and write them to a csv file
python3 util/parse_xml.py

# create a directory for the small generated test data
mkdir data/small

# inject the search string into the dataset
# configure the selectivities to your needs
python3 util/dblp_generator.py

# multiply the dataset so it gets larger
python3 util/multiply.py ./data/small/dblp_0.01_data.csv ./data/small/dblp_0.32_data.csv

# remove special characters from the dataset
python3 util/remove_special_characters.py ./data/dblp_0.01_data.csv ./data/dblp_0.32_data.csv

# create file containing the search string
echo "performance optimization of gpu string processing" > data/search.csv
```

### Execute single test

```bash
# create benchmarking directory
mkdir benchmark

# copy benchmarking programs to benchmark directory for easier configuration
cp equals/* benchmark

# change into the benchmark directory to shorten commands
cd benchmark

# insert the correct configuration into the load script
# modify these parameters to your needs
sed 's:#define FILENAME :#define FILENAME "../data/dblp_0.01_data.csv":g ; s:#define LINE_COUNT :#define LINE_COUNT 22741465:g ; s:#define SEARCH_FILENAME "":#define SEARCH_FILENAME "../data/search.csv":g' load_db.cu > load_db_0.01.cu

# compile the load script
nvcc load_db_0.01.cu -o load_db_0.01

# create directory for the memory mapped files
mkdir mmdb

# execute the load script
./load_db_0.01

# insert the correct configuration into the actual benchmark script
# modify these parameters to your needs
sed 's:#define BLOCK_SIZE :#define BLOCK_SIZE 256:g ; s:#define KERNEL_NAME :#define KERNEL_NAME bufferKernel:g' equals.cu > prefix_buffer_256.cu

# compile the benchmark script
nvcc prefix_buffer_256.cu -o prefix_buffer_256

# execute the benchmark script
./prefix_buffer_256 1000
```

### Further tests

Executing the tests for the regular expression matcher works similarly.
Please make sure that you use the correct `load_db.cu` script.

You can also execute automated tests with the script `util/benchmark_many_parameters.py`.