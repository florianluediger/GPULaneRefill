#include <iostream>
#include <ctime>
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"

#define BLOCK_SIZE              // e.g. {32,64,96,128,160,192,224,256,384,512,640,786}
#define KERNEL_NAME             // e.g. {naiveKernel,bufferKernel,unrollKernel}
#define COMPARISON_OPERATOR <   // != for equality test; < for prefix test

#define ALL_LANES 0xffffffff

__global__
void unrollKernel(int *character_offset, char *book_content, char *search_string, int search_string_length, int line_count, int *number_of_matches) {
    bool active = 1;            // gets set to 0 when selection fails
    bool flush_pipeline = 0;    // break computation when every line is finished/inactive
    unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x);      // global index of the current iteration
    unsigned step = (blockDim.x * gridDim.x);                           // offset for the next element to be computed
    unsigned current_element = 0;       // current element to be computed
    unsigned buffercount = 0;
    unsigned warpid = (threadIdx.x / 32);
    unsigned bufferbase = (warpid * 32);
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));
    __shared__ int character_index_divergence_buffer[BLOCK_SIZE];
    __shared__ int current_element_divergence_buffer[BLOCK_SIZE];

    while(!flush_pipeline) {
        current_element = loop_var;
        active = current_element < line_count;
        flush_pipeline = !__ballot_sync(ALL_LANES, active);

        // If the lengths of the strings don't match, the string can be discarded immediately
        if (active && character_offset[current_element+1] - character_offset[current_element] - 1 COMPARISON_OPERATOR search_string_length)
            active = false;

        int character_index = 0;
        unsigned activemask = __ballot_sync(ALL_LANES, active);
        unsigned numactive = __popc(activemask);
        unsigned scan;
        int numRemaining;
        int numRefill;
        unsigned buf_ix;
        unsigned bail_out_threshold = flush_pipeline ? 0 : 25;

        while(buffercount + numactive > bail_out_threshold) {
            if (numactive < 25 && buffercount) {
                numRefill = min(32 - numactive, buffercount);
                numRemaining = buffercount - numRefill;

                scan = __popc(~activemask & prefixlanes);

                if (!active && scan < buffercount) {
                    buf_ix = numRemaining + scan + bufferbase;
                    character_index = character_index_divergence_buffer[buf_ix];
                    current_element = current_element_divergence_buffer[buf_ix];
                    active = true;
                }

                buffercount -= numRefill;
            }
#pragma unroll
            for (int u = 0; u < 3; u++) {
                if (active && book_content[character_index + character_offset[current_element]] != search_string[character_index])
                    active = false;
            
                character_index++;

                if (character_index == search_string_length) {
                    if(active)
                        atomicAdd(number_of_matches, 1);
                    active = false;
                }
            }

            activemask = __ballot_sync(ALL_LANES, active);
            numactive = __popc(activemask);
        }

        if(numactive > 0) {
            scan = __popc(activemask & prefixlanes);
            buf_ix = bufferbase + buffercount + scan;

            if(active) {
                character_index_divergence_buffer[buf_ix] = character_index;
                current_element_divergence_buffer[buf_ix] = current_element;
            }

            buffercount += numactive;
            active = false;
        }
        
        loop_var += step;
    }
}

__global__
void bufferKernel(int *character_offset, char *book_content, char *search_string, int search_string_length, int line_count, int *number_of_matches) {
    bool active = 1;            // gets set to 0 when selection fails
    bool flush_pipeline = 0;    // break computation when every line is finished/inactive
    unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x);      // global index of the current iteration
    unsigned step = (blockDim.x * gridDim.x);                           // offset for the next element to be computed
    unsigned current_element = 0;       // current element to be computed
    unsigned buffercount = 0;
    unsigned warpid = (threadIdx.x / 32);
    unsigned bufferbase = (warpid * 32);
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));
    __shared__ int character_index_divergence_buffer[BLOCK_SIZE];
    __shared__ int current_element_divergence_buffer[BLOCK_SIZE];

    while(!flush_pipeline) {
        current_element = loop_var;
        active = current_element < line_count;
        flush_pipeline = !__ballot_sync(ALL_LANES, active);

        // If the lengths of the strings don't match, the string can be discarded immediately
        if (active && character_offset[current_element+1] - character_offset[current_element] - 1 COMPARISON_OPERATOR search_string_length)
            active = false;

        int character_index = 0;
        unsigned activemask = __ballot_sync(ALL_LANES, active);
        unsigned numactive = __popc(activemask);
        unsigned scan;
        int numRemaining;
        int numRefill;
        unsigned buf_ix;
        unsigned bail_out_threshold = flush_pipeline ? 0 : 25;

        while(buffercount + numactive > bail_out_threshold) {
            if (numactive < 25 && buffercount) {
                numRefill = min(32 - numactive, buffercount);
                numRemaining = buffercount - numRefill;

                scan = __popc(~activemask & prefixlanes);

                if (!active && scan < buffercount) {
                    buf_ix = numRemaining + scan + bufferbase;
                    character_index = character_index_divergence_buffer[buf_ix];
                    current_element = current_element_divergence_buffer[buf_ix];
                    active = true;
                }

                buffercount -= numRefill;
            }

            if (active && book_content[character_index + character_offset[current_element]] != search_string[character_index])
                active = false;
        
            character_index++;

            if (character_index == search_string_length) {
                if(active)
                    atomicAdd(number_of_matches, 1);
                active = false;
            }

            activemask = __ballot_sync(ALL_LANES, active);
            numactive = __popc(activemask);
        }

        if(numactive > 0) {
            scan = __popc(activemask & prefixlanes);
            buf_ix = bufferbase + buffercount + scan;

            if(active) {
                character_index_divergence_buffer[buf_ix] = character_index;
                current_element_divergence_buffer[buf_ix] = current_element;
            }

            buffercount += numactive;
            active = false;
        }
        
        loop_var += step;
    }
}

__global__
void naiveKernel(int *character_offset, char *book_content, char *search_string, int search_string_length, int line_count, int *number_of_matches) {
    bool active = 1;                // gets set to 0 when selection fails
    bool flush_pipeline = 0;        // break computation when every line is finished/inactive
    unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x);  // global index of the element to be computed
    unsigned step = (blockDim.x * gridDim.x);                       // offset for the next element to be computed

    while(!flush_pipeline) {
        active = loop_var < line_count;
        flush_pipeline = !__ballot_sync(ALL_LANES, active);

        // If the lengths of the strings don't match, the string can be discarded immediately
        if (active && character_offset[loop_var+1] - character_offset[loop_var] - 1 COMPARISON_OPERATOR search_string_length) active = false;

        int character_index = 0;
        while(__any_sync(ALL_LANES, active) && character_index < search_string_length) {
            if (active && book_content[character_index + character_offset[loop_var]] != search_string[character_index])
                active = false;
        
            character_index++;
        }

        if (active)
            atomicAdd(number_of_matches, 1);
        
        loop_var += step;
    }
}

int main(int argc, char *argv[]) {
    int grid_size = atoi(argv[1]);

    // ### Create pointers to memory mapped files ### 

    std::clock_t start_import = std::clock();
    
    int *character_offset;
    character_offset = (int *)map_memory_file("mmdb/book_line_offset");
    int *book_content;
    book_content = (int *)map_memory_file("mmdb/book_content");
    int *book_meta;
    book_meta = (int *)map_memory_file("mmdb/book_meta");
    int line_count = book_meta[0];
    int character_count = book_meta[1];
    char *search_string;
    search_string = (char *)map_memory_file("mmdb/search_string");
    int search_string_length = strlen(search_string);

    std::clock_t stop_import = std::clock();

    // ### Copy the data into the GPU's memory ###

    cudaDeviceSynchronize();
    
    std::clock_t start_cuda_malloc = std::clock();

    int *d_character_offset, *d_number_of_matches;
    char *d_book_content;
    char *d_search_string;
    cudaMalloc((void **)&d_character_offset, sizeof(int) * (line_count + 1));
    cudaMalloc((char **)&d_book_content, sizeof(char) * character_count);
    cudaMalloc((char **)&d_search_string, sizeof(char) * search_string_length);
    cudaMalloc((void **)&d_number_of_matches, sizeof(int));
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_malloc = std::clock();
    std::clock_t start_cuda_memcpy = std::clock();

    int *number_of_matches;
    number_of_matches = new int(0);

    cudaMemcpy(d_character_offset, character_offset, sizeof(int) * (line_count + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_book_content, book_content, sizeof(char) * character_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_search_string, search_string, sizeof(char) * search_string_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_number_of_matches, number_of_matches, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    std::clock_t stop_cuda_memcpy = std::clock();

    // ### Execute the desired tasks on the GPU ###

    std::clock_t start_test_kernel = std::clock();
    KERNEL_NAME<<<grid_size, BLOCK_SIZE>>>(d_character_offset, d_book_content, d_search_string, search_string_length, line_count, d_number_of_matches);
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
    cudaFree(d_search_string);
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
