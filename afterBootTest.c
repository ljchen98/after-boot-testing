#define _GNU_SOURCE
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <sched.h>
#include <time.h>

#define L1CACHE_SIZE 256      // L1 Cache 的大小 (in Bytes)
#define L2CACHE_SIZE 2*1024   // L2 Cache 的大小 (in Bytes)
#define MEMORY_SIZE 2*1024  // RAM 的大小 (in Bytes)
#define DEFINED_TYPE u_int8_t // 访存最小单位的类型 (u_int8_t, u_int16_t, etc.)

#define TYPE_WIDTH (sizeof(DEFINED_TYPE) << 3)  // 计算出访存最小单位的宽度 (in bits)
#define TYPE_RANGE (1 << TYPE_WIDTH)            // 计算出访存最小单位的范围

// 单线程访存：内存范围内地址的写与读 -- 背靠背写读
void singleCoreAllAddrWR_B2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    size_t core_id;
    core_id = sched_getcpu();
    printf("[核%lu写读] 单线程访存：内存范围内地址的写与读 -- 背靠背写读\n", core_id);
    for (size_t i = 0; i < *n; i++) {
        arr[i] = i % TYPE_RANGE;
        assert(arr[i] == i % TYPE_RANGE);
    }
    printf("[\033[33mDone\033[0m] \n");
    printf("----------------------------------------------------------------------------- \n");
}

// 单线程访存：内存范围内地址的写与读 -- 非背靠背写读
void singleCoreAllAddrWR_nonB2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    size_t core_id;
    core_id = sched_getcpu();
    printf("[核%lu写读] 单线程访存：内存范围内地址的写与读 -- 非背靠背写读\n", core_id);
    for (size_t i = 0; i < *n; i++) {
        arr[i] = i % TYPE_RANGE;
    }
    for (size_t i = 0; i < *n; i++) {
        assert(arr[i] == i % TYPE_RANGE);
    }
    printf("[\033[33mDone\033[0m] \n");
    printf("----------------------------------------------------------------------------- \n");
}

// 单线程访存：分别单独使用每个数据位写与读 -- 背靠背写读
void singleCoreAllBitsWR_B2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    size_t core_id;
    core_id = sched_getcpu();
    printf("[核%lu写读] 单线程访存：分别单独使用每个数据位写与读 -- 背靠背写读 \n", core_id);
    for (size_t i = 0; i < *n; i++) {
        for (size_t j = 0; j < TYPE_WIDTH; j++) {
            arr[i] = 1 << j;
            assert(arr[i] == 1 << j);
        }
    }
    printf("[\033[33mDone\033[0m] \n");
    printf("----------------------------------------------------------------------------- \n");
}

// 单线程访存：分别单独使用每个数据位写与读 -- 非背靠背写读
void singleCoreAllBitsWR_nonB2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    size_t core_id;
    core_id = sched_getcpu();
    printf("[核%lu写读] 单线程访存：分别单独使用每个数据位写与读 -- 非背靠背写读 \n", core_id);
    for (size_t j = 0; j < TYPE_WIDTH; j++) {
        for (size_t i = 0; i < *n; i++) {
            arr[i] = 1 << j;
        }
        for (size_t i = 0; i < *n; i++) {
            assert(arr[i] == 1 << j);
        }
    }
    printf("[\033[33mDone\033[0m] \n");
    printf("----------------------------------------------------------------------------- \n");
}

// 多线程访存：内存范围内地址的先写后读 -- 背靠背写读
void multiCoreAllAddrWR_B2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    printf("[多核写读] 多线程访存：内存范围内地址的写与读 -- 背靠背写读 \n");
    size_t thread_count = omp_get_max_threads();
    // srand((unsigned)time(NULL));
    // printf("Mem Size: MEMORY_SIZE\n", *n);
    for (size_t i = 0; i < *n; i++){
        for (size_t j = 0; j < thread_count; j++)
            for (size_t k = 0; k < thread_count; k++){
                if (j == k)
                    continue;
                // size_t core_w, core_r;
                // DEFINED_TYPE randNum = rand() % TYPE_RANGE;
                DEFINED_TYPE wrNum = (i + j * thread_count + k) % TYPE_RANGE;
                #pragma omp parallel
                {
                    size_t my_rank = omp_get_thread_num();
                    if (my_rank == j){
                        // core_w = sched_getcpu();
                        arr[i] = wrNum;
                    }
                }
                #pragma omp parallel
                {
                    size_t my_rank = omp_get_thread_num();
                    if (my_rank == k){
                        // core_r = sched_getcpu();
                        assert(arr[i] == wrNum);
                    }
                }
                // printf("\t%lu:%luw%lur", i, core_w, core_r);
            }
        // printf("\t %lu", i);
    }
    printf("[\033[33mDone\033[0m]  \n");
    printf("----------------------------------------------------------------------------- \n");
}

// 多线程访存：内存范围内地址的先写后读 -- 非背靠背写读
void multiCoreAllAddrWR_nonB2B (DEFINED_TYPE *arr, size_t *n){
    printf("\n----------------------------------------------------------------------------- \n");
    printf("[多核写读] 多线程访存：内存范围内地址的先写后读 -- 非背靠背写读 \n");
    size_t thread_count = omp_get_max_threads();
    for (size_t i = 0 ; i < thread_count; i++){
        for (size_t j = 0 ; j < thread_count; j++){
            if (i == j)
                continue;
            size_t core_w, core_r;
            #pragma omp parallel
            {
                size_t my_rank = omp_get_thread_num();
                if (my_rank == i){
                    core_w = sched_getcpu();
                    for (size_t k = 0; k < *n; k++) {
                        arr[k] = k % TYPE_RANGE;
                    }
                }
            }
            #pragma omp parallel
            {
                size_t my_rank = omp_get_thread_num();
                if (my_rank == j){
                    core_r = sched_getcpu();
                    for (size_t k = 0; k < *n; k++) {
                        assert(arr[k] == k % TYPE_RANGE);
                    }
                }
            }
            printf("\t(核%lu写, 核%lu读)", core_w, core_r);
        }
        printf("\n");
    }
    printf("[\033[33mDone\033[0m] \n");
    printf("----------------------------------------------------------------------------- \n");
}


int main( int argc, char* argv[] ) {

    if (argc != 2) {
        printf("usage: %s <MEMORY_WR_SIZE_in_byte>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t memory_wr_size = atoi(argv[1]);
    DEFINED_TYPE *arr = (u_int8_t *)malloc(memory_wr_size);
    size_t n = memory_wr_size / sizeof(DEFINED_TYPE);

    // 单线程访存：内存范围内地址的写与读 -- 背靠背写读
    singleCoreAllAddrWR_B2B(arr, &n);

    // 单线程访存：内存范围内地址的写与读 -- 非背靠背写读
    singleCoreAllAddrWR_nonB2B(arr, &n);

    // 单线程访存：分别单独使用每个数据位写与读 -- 背靠背写读
    singleCoreAllBitsWR_B2B(arr, &n);

    // 单线程访存：分别单独使用每个数据位写与读 -- 非背靠背写读
    singleCoreAllBitsWR_nonB2B(arr, &n);

    // 多线程访存：内存范围内地址的写与读 -- 背靠背写读
    multiCoreAllAddrWR_B2B(arr, &n);

    // 多线程访存：内存范围内地址的写与读 -- 非背靠背写读
    multiCoreAllAddrWR_nonB2B(arr, &n);



    printf("[\033[33mAll test done!\033[0m]\n");
    return EXIT_SUCCESS;
}



