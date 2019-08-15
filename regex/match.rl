#include <iostream>
#include <ctime>
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"

#define BLOCK_SIZE              // e.g. {32,64,96,128,160,192,224,256,384,512,640,786}
#define KERNEL_NAME             // e.g. {naiveKernel,bufferKernel,unrollKernel}
#define STARTS_WITH_ANY true	// true if you allow any number of arbitrary symbols at the beginning of your string e.g. .*

/* This is just to prevent compiler errors depending on using flat or table option */
__device__ static const char *_machine_key_offsets = 0;
__device__ static const char *_machine_single_lengths = 0;
__device__ static const char *_machine_range_lengths = 0;

// __device__ static const char *_machine_indicies = 0;
// __device__ static const char *_machine_key_spans = 0;

%%{
    machine machine;
    write data;

	main:= '';      // insert your regular expression here according to the ragel syntax
}%%

struct str_t {
    char* start;
    char* end;
};

__inline__ __device__ bool cmpLike ( char c, char l ) {
    return ( c == l ) || ( l == '_' );
}

// Author of this function: Henning Funke
__device__ bool stringLikeCheck ( str_t string, str_t like ) {
    char *sPos, *lPos, *sTrace, *lTrace;
    char *lInStart = like.start;
    char *lInEnd   = like.end;
    char *sInStart = string.start;
	char *sInEnd   = string.end;

    // prefix 
    if ( *like.start != '%' ) { 
        sPos = string.start;
        lPos = like.start;
        for ( ; lPos < like.end && sPos < string.end && (*lPos) != '%'; ++lPos, ++sPos ) {
            if ( !cmpLike ( *sPos, *lPos ) )
                return false;
        }
        lInStart = lPos; 
        sInStart = sPos; 
    }
    
    // suffix 
    if ( *(like.end-1) != '%' ) {
        sPos = string.end-1;
        lPos = like.end-1;
        for ( ; lPos >= like.start && sPos >= string.start && (*lPos) != '%'; --lPos, --sPos ) {
            if ( !cmpLike ( *sPos, *lPos ) )
                return false;
        }
        lInEnd = lPos;
        sInEnd = sPos+1; // first suffix char 
    }

    // infixes 
    if ( lInStart < lInEnd ) {
        lPos = lInStart+1; // skip '%'
        sPos = sInStart;
        while ( sPos < sInEnd && lPos < lInEnd ) { // loop 's' string
            lTrace = lPos;
            sTrace = sPos;
            while ( cmpLike ( *sTrace, *lTrace ) && sTrace < sInEnd ) { // loop infix matches
                ++lTrace;
                if ( *lTrace == '%' ) {
                    lPos = ++lTrace;
                    sPos = sTrace;
                    break;
                }
                ++sTrace; 
            }
            ++sPos;
        }
    }
    return lPos >= lInEnd;
}

__device__ int singleDfaStepTable(int cs, char* p) {
	int _klen;
	unsigned int _trans;
	const char *_keys;

	_keys = _machine_trans_keys + _machine_key_offsets[cs];
	_trans = _machine_index_offsets[cs];

	_klen = _machine_single_lengths[cs];
	if (_klen > 0)
	{
		const char *_lower = _keys;
		const char *_mid;
		const char *_upper = _keys + _klen - 1;
		while (1)
		{
			if (_upper < _lower) {
				_keys += _klen;
				_trans += _klen;
				break;
			}

			_mid = _lower + ((_upper - _lower) >> 1);

			if ((*p) < *_mid)
				_upper = _mid - 1;
			else if ((*p) > *_mid)
				_lower = _mid + 1;
			else
			{
				_trans += (unsigned int)(_mid - _keys);
				break;
			}
		}
	}

	_klen = _machine_range_lengths[cs];
	if (_klen > 0)
	{
		const char *_lower = _keys;
		const char *_mid;
		const char *_upper = _keys + (_klen << 1) - 2;
		while (1)
		{
			if (_upper < _lower) {
				_trans += _klen;
				break;
			}

			_mid = _lower + (((_upper - _lower) >> 1) & ~1);

			if ((*p) < _mid[0])
				_upper = _mid - 2;
			else if ((*p) > _mid[1])
				_lower = _mid + 2;
			else
			{
				_trans += (unsigned int)((_mid - _keys) >> 1);
				break;
			}
		}
	}

	return _machine_trans_targs[_trans];
}

__device__ int singleDfaStepFlat(int cs, char* p) {
	int _slen;
	int _trans;
	const char *_keys;
	const char *_inds;

	_keys = _machine_trans_keys + (cs<<1);
	_inds = _machine_indicies + _machine_index_offsets[cs];

	_slen = _machine_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	return _machine_trans_targs[_trans];
}

__global__ void dogQCKernel(int *character_offset, char *book_content, int line_count, char* regex_start, char* regex_end, int *number_of_matches) {
	bool active = 1;											   // gets set to 0 when selection fails
	bool flush_pipeline = 0;									   // break computation when every line is finished/inactive
	unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x); // global index of the element to be computed
	unsigned step = (blockDim.x * gridDim.x);					   // offset for the next element to be computed
	str_t current_string;
	str_t regex_string;

	regex_string.start = regex_start;
	regex_string.end = regex_end;

	while (!flush_pipeline)
	{
		active = loop_var < line_count;
		flush_pipeline = !__ballot_sync(ALL_LANES, active);

		if (active) {
			current_string.start = book_content + character_offset[loop_var];
			current_string.end = book_content + character_offset[loop_var + 1];

			if (stringLikeCheck(current_string, regex_string))
				atomicAdd(number_of_matches, 1);
			else
				active = false;
		}
		loop_var += step;
	}
}

__global__ void bufferKernel(int *character_offset, char *book_content, int line_count, int *number_of_matches)
{
	bool active = 1;											   // gets set to 0 when selection fails
	bool flush_pipeline = 0;									   // break computation when every line is finished/inactive
	unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x); // global index of the element to be computed
	unsigned step = (blockDim.x * gridDim.x);					   // offset for the next element to be computed

	unsigned buffercount = 0;
	unsigned warpid = (threadIdx.x / 32);
	unsigned bufferbase = (warpid * 32);
	unsigned warplane = (threadIdx.x % 32);
	unsigned prefixlanes = (0xffffffff >> (32 - warplane));
	__shared__ char* p_divergence_buffer[BLOCK_SIZE];
	__shared__ char* pe_divergence_buffer[BLOCK_SIZE];
	__shared__ int cs_divergence_buffer[BLOCK_SIZE];

	while (!flush_pipeline)
	{
		active = loop_var < line_count;
		flush_pipeline = !__ballot_sync(ALL_LANES, active);

		char *p = book_content + character_offset[loop_var];
		char *pe = book_content + character_offset[loop_var + 1];

		int cs = machine_start;

		if (p == pe)
			active = false;

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
                    p = p_divergence_buffer[buf_ix];
					pe = pe_divergence_buffer[buf_ix];
					cs = cs_divergence_buffer[buf_ix];
                    active = true;
                }

                buffercount -= numRefill;
			}

			if (active) {
				cs = singleDfaStepFlat(cs, p);
				// cs = singleDfaStepTable(cs, p);

				if (cs == 0 && !STARTS_WITH_ANY)		// invalid state reached
					active = false;
			}
				
			p++;

			if (active && p == pe) {		// string completely processed
				if (cs >= machine_first_final)
					atomicAdd(number_of_matches, 1);
				active = false;
			}

			activemask = __ballot_sync(ALL_LANES, active);
            numactive = __popc(activemask);
		}

		if (numactive > 0) {
			scan = __popc(activemask & prefixlanes);
            buf_ix = bufferbase + buffercount + scan;

            if(active) {
                p_divergence_buffer[buf_ix] = p;
				pe_divergence_buffer[buf_ix] = pe;
				cs_divergence_buffer[buf_ix] = cs;
            }

            buffercount += numactive;
            active = false;
		}

		loop_var += step;
	}
}

__global__ void matchKernel(int *character_offset, char *book_content, int line_count, int *number_of_matches)
{
	bool active = 1;											   // gets set to 0 when selection fails
	bool flush_pipeline = 0;									   // break computation when every line is finished/inactive
	unsigned loop_var = ((blockIdx.x * blockDim.x) + threadIdx.x); // global index of the element to be computed
	unsigned step = (blockDim.x * gridDim.x);					   // offset for the next element to be computed

	while (!flush_pipeline)
	{
		active = loop_var < line_count;
		flush_pipeline = !__ballot_sync(ALL_LANES, active);

		char *p = book_content + character_offset[loop_var];
		char *pe = book_content + character_offset[loop_var + 1];

		int cs = machine_start;

		if (p == pe)
			active = false;

		while(__any_sync(ALL_LANES, active)) {
			if (active) {
				cs = singleDfaStepFlat(cs, p);
				// cs = singleDfaStepTable(cs, p);

				p++;

				if (p == pe)		// string completely processed
					active = false;

				if (cs == 0 && !STARTS_WITH_ANY)		// invalid state reached
					active = false;
			}
		}

		if (cs >= machine_first_final) {
			active = true;
			atomicAdd(number_of_matches, 1);
		}
		else {
			active = false;
		}

		loop_var += step;
	}
}

int main(int argc, char *argv[])
{
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

	// ### Prepare some data for the dogQCKernel ###

	char re[] = "performance optimization of gpu%";
	char *d_regular_expression;
	cudaMalloc((char **)&d_regular_expression, sizeof(char) * strlen(re));
	cudaMemcpy(d_regular_expression, re, sizeof(char) * strlen(re), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// ### Execute the desired tasks on the GPU ###

	std::clock_t start_test_kernel = std::clock();
	bufferKernel<<<grid_size, BLOCK_SIZE>>>(d_character_offset, d_book_content, line_count, d_number_of_matches);
	// dogQCKernel<<<grid_size, BLOCK_SIZE>>>(d_character_offset, d_book_content, line_count, d_regular_expression, d_regular_expression + strlen(re), d_number_of_matches);
	
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
