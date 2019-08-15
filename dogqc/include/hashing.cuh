// Author of this file: Henning Funke

#include "util.cuh"




__device__ uint64_t hash( uint64_t a) {
    a -= (a<<6);
    a ^= (a>>17);
    a -= (a<<9);
    a ^= (a<<4);
    a -= (a<<3);
    a ^= (a<<10);
    a ^= (a>>15);
    return a;
}

__device__ uint64_t hash( uint32_t a) {
    return hash ( (uint64_t)a );
}


template <typename T>
struct unique_ht {
    uint32_t key;
    T payload;
};


// intialize an array as used e.g. for join hash tables
template <typename T>
__global__ void initUniqueHT ( unique_ht<T>* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].key = 0xffffffff;
    }
}


template <typename T>
__device__ void hashBuildUnique ( unique_ht<T>* hash_table, int ht_size, uint32_t key, T& payload ) {
    int org_location = hash(key) % ht_size;
    uint32_t location = org_location;
    unique_ht<T>* elem;

    while ( true )  {
        elem = &(hash_table[location]);
        unsigned int probe_key = atomicCAS( (unsigned int*) &(elem->key), 0xffffffff, (unsigned int)key );
        if(probe_key == 0xffffffff) {
            elem->payload = payload;
	    return;
        }
        location = (location + 1) % ht_size;
        if(location == org_location)
            return;
    }
}


template <typename T>
__device__ bool hashProbeUnique ( unique_ht<T>* hash_table, int ht_size, uint32_t key, T& payload ) {
    int org_location = hash(key) % ht_size;
    uint32_t location = org_location;
    unique_ht<T>* elem;

    while ( true )  {
        elem = &(hash_table[location]);
        unsigned int probeKey = elem->key;
        if(probeKey == key) {
            payload = elem->payload;
            return true;
        }
        if(probeKey == 0xffffffff) {
            return false;
        }
        location = (location + 1) % ht_size;
        if(location == org_location)
            return false;
    }
}


// return value indicates whether the calling thread should execute the section 
template<typename L>
__device__ bool oneTimeLockEnter ( L* lock, L initVal, L workingVal ) {
    L lockState = atomicCAS( lock, initVal, workingVal );
    __threadfence();
    return lockState == initVal;
}


template<typename L>
__device__ void oneTimeLockDone ( L* lock, L doneVal ) {
    __threadfence();
    atomicExch( lock, doneVal );
}


template<typename L>
__device__ void oneTimeLockWait ( L* lock, L initVal, L workingVal ) {
    int i = 0;
    // any lane wokingVal or initVal
    while ( ( *lock == workingVal) || ( *lock == initVal) ) {
        i++;
        // the following two lines are critical
        // otherwise the code will be broken by compiler optimizations
        if(i > 1000000) 
            printf ( "loop count high: %i\n" );
    }
    __threadfence();
}


struct multi_ht {
    uint32_t key;
    uint32_t offset;
    uint32_t count;
    uint32_t insertcount;
};


// intialize an array as used e.g. for join hash tables
__global__ void initMultiHT ( multi_ht* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].key = 0xffffffff;
	ht[i].offset = 0xffffffff;
	ht[i].count = 0;
	ht[i].insertcount = 0;
    }
}


// join hash insert: count the number of matching elements
__device__ int hashCountMulti ( multi_ht* ht, int32_t ht_size, uint32_t key ) {

    int org_location = hash(key) % ht_size;
    uint32_t location = org_location;

    while( true )  {

        multi_ht* entry = &ht[location];
	uint32_t x = entry->key;
        
        // bucket empty, try to allocate for key
        if ( entry->key == 0xffffffff ) {
            x = atomicCAS( &(entry->key), 0xffffffff, key);
            if ( x == 0xffffffff ) {
                // hash table position x now stores the desired key
                x = key;
            }
        }
        // reached right bucket
        if ( x == key ) {
            atomicAdd ( &(entry->count), 1);
            return 1;
        }
        
        location = (location + 1) % ht_size;

        // exit if right bucket can not be found
        if(location == org_location)
            return -1;
    }
}


// join hash insert: insert elements
template <typename T>
__device__ void hashInsertMulti ( multi_ht* ht, T* payload, int* range_offset, int32_t ht_size, uint32_t key, T& payl ) {

    int org_location = hash(key) % ht_size;
    uint32_t location = org_location;

    while( true )  {

        multi_ht* entry = &ht [ location ];
        
        // bucket has our key
        if ( entry->key == key ) {

             // offset of matching tuple range is allocated by one thread
             uint32_t* address = &(entry->offset);

	     // allocate offset only once
             if ( oneTimeLockEnter ( address, 0xffffffff, 0xfffffffe ) ) {
                 // initialize insertion counter
                 entry->insertcount = entry->count;
		 // get base offset (shared across tuples with same key)
                 uint32_t baseOffset = atomicAdd ( range_offset, entry->count + 1 );
                 // release lock and write base offset for key
                 oneTimeLockDone ( address, baseOffset );
             }
            
	     // make sure initialization is finished 
	     oneTimeLockWait ( address, 0xffffffff, 0xfffffffe );
             
	     // write key into output range
             uint32_t baseOffset = *address;
             uint32_t tupleOffset = atomicSub ( &(entry->insertcount), 1) - 1;
	     payload [ tupleOffset  + baseOffset ] = payl;
	     return;
        }

        location = (location + 1) % ht_size;

        // exit if right bucket can not be found
        if(location == org_location)
	    return;
    }
}


// join hash probe
__device__ bool hashProbeMulti ( multi_ht* ht, uint32_t ht_size, uint32_t key, int& offset, int& end ) {
    uint32_t location = hash(key) % ht_size;
    int num_probes = 0;
    offset = 0xffffffff;
    uint32_t x;

    while( num_probes <= ht_size )  {
    
        multi_ht* entry = &( ht [ location ] );

        x = entry->key;
        num_probes++;

        if(x == key) {
            offset = entry->offset;
            end = offset + entry->count;
            return true;
        }

        if(x == 0xffffffff) {
            return false;
        }

        location = (location + 1) % ht_size;

    }
    // exit if hash table is full and right bucket can not be found
    return false;
}


template <typename T>
struct agg_ht {
    uint32_t lock;
    uint64_t hash;
    T payload;
};


template <typename T>
__global__ void initAggHT ( agg_ht<T>* ht, int32_t num ) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
	ht[i].lock = 0xffffffff;
    }
}


// returns candidate bucket
template <typename T>
__device__ int hashAggregateGetBucket ( agg_ht<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, T& payl ) {
    while ( numLookups < ht_size )  {
        int location = ( grouphash + numLookups ) % ht_size;
        agg_ht<T>& entry = ht [ location ];
	numLookups++;
        // bucket is empty
        if ( entry.lock == 0xffffffff ) {
	    // initialize bucket only once
            if ( oneTimeLockEnter ( &entry.lock, 0xffffffff, 0xfffffffe ) ) {
                entry.payload = payl;
		entry.hash = grouphash;
                oneTimeLockDone ( &entry.lock, 0u);
	        return location;
            }
	}
	oneTimeLockWait ( &entry.lock, 0xffffffff, 0xfffffffe );
	if ( entry.hash == grouphash ) {
            return location;
	}
    }
    printf ( "hash table full\n" );
    return -1;
}
