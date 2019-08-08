#include <iostream>
#include <ctime>
#include "../dog-qc/dogqc/include/csv.h"
#include "../dog-qc/dogqc/include/util.h"
#include "../dog-qc/dogqc/include/mappedmalloc.h"

#define BLOCK_SIZE 1
#define ALL_LANES 0xffffffff

%%{
    machine foo;
    write data;
}%%

__global__
void matchKernel(int *character_offset, char *book_content, int line_count, int *number_of_matches) {
    bool active = 1;            // gets set to 0 when selection fails
    bool flush_pipeline = 0;     // break computation when every line is finished/inactive
    unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x);       // global index of the element to be computed
    unsigned step = (blockDim.x * gridDim.x);                           // offset for the next element to be computed

    while(!flush_pipeline) {
        active = loop_var < line_count;
        flush_pipeline = !__ballot_sync(ALL_LANES, active);
       
        char *p = book_content + character_offset[loop_var];
        char *pe = book_content + character_offset[loop_var + 1];

        int cs;
        
        if (active) {
            %%{
                main:= any* 'performance optimization of gpu' any*;
                write init;
                write exec;
            }%%
        }

        if (cs < foo_first_final)
            active = false;

        if (active)
            atomicAdd(number_of_matches, 1);
        
        loop_var += step;
    }
}

int main(int argc, char *argv[]) {
    int grid_size = atoi(argv[1]);

    // ### Create pointers to memory mapped files### 

    std::clock_t start_import = std::clock();
    
    int *character_offset;
    character_offset = (int *)map_memory_file("mmdb/book_line_offset");
    int *book_content;
    book_content = (int *)map_memory_file("mmdb/book_content");
    int *book_meta;
    book_meta = (int *)map_memory_file("mmdb/book_meta");
    int line_count = book_meta[0];
    int character_count = book_meta[1];

    std::clock_t stop_import = std::clock();

    // ### Copy the data into the GPU's memory ###

    cudaDeviceSynchronize();
    
    std::clock_t start_cuda_malloc = std::clock();

    int *d_character_offset, *d_number_of_matches;
    char *d_book_content;
    cudaMalloc((void **)&d_character_offset, sizeof(int) * (line_count + 1));
    cudaMalloc((char **)&d_book_content, sizeof(char) * character_count);
    cudaMalloc((void **)&d_number_of_matches, sizeof(int));
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_malloc = std::clock();
    std::clock_t start_cuda_memcpy = std::clock();

    int *number_of_matches;
    number_of_matches = new int(0);

    cudaMemcpy(d_character_offset, character_offset, sizeof(int) * (line_count + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_book_content, book_content, sizeof(char) * character_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_of_matches, number_of_matches, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_memcpy = std::clock();

    // ### Execute the desired tasks on the GPU ###

    std::clock_t start_test_kernel = std::clock();
    matchKernel<<<grid_size, BLOCK_SIZE>>>(d_character_offset, d_book_content, line_count, d_number_of_matches);
    cudaDeviceSynchronize();
    std::clock_t stop_test_kernel = std::clock();

    // ### Retrieve results from the GPU ###

    std::clock_t start_cuda_get_result = std::clock();

    cudaMemcpy(number_of_matches, d_number_of_matches, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_get_result = std::clock();

    printf("Number of matches is: %d\n", *number_of_matches);

    // ### Free the memory on the GPU ###

    std::clock_t start_cuda_free = std::clock();

    cudaFree(d_character_offset);
    cudaFree(d_book_content);
    cudaFree(d_number_of_matches);
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_free = std::clock();

    // std::cout << (stop_test_kernel - start_test_kernel) / (double)(CLOCKS_PER_SEC / 1000) << std::endl;

    std::cout << "grid size: " << grid_size << std::endl;
    std::cout << "block_size: " << BLOCK_SIZE << std::endl;
    std::cout << "import: " << (stop_import - start_import) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "cuda malloc: " << (stop_cuda_malloc - start_cuda_malloc) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "cuda memcopy: " << (stop_cuda_memcpy - start_cuda_memcpy) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "test kernel: " << (stop_test_kernel - start_test_kernel) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "retrieve result: " << (stop_cuda_get_result - start_cuda_get_result) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    std::cout << "cuda free: " << (stop_cuda_free - start_cuda_free) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    return 0;
}
