#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

uint32_t basecheck[] = {0x7b8c74ad, 0x99c4b23a, 0xa08a5b6f, 0xf5d208b1, 0x7210b57f, 0x33ece712, 0xfa660c0e, 0xa4c55a85, 0xb6f4af98, 0xcb1dbb65, 0xc772426d, 0x8a393361, 0x5e0e6c2c, 0x3d38c6d2, 0xe073c5f4, 0x74b0cddd, 0xbbb4468a, 0xbf046823, 0x5ae0ca06, 0xc200c70f, 0x6edabd3b, 0x17a24bdf, 0x6da49b4, 0x85770bd2, 0x558f8e7, 0x2db779ba, 0x6e37a6a0, 0x36f950cb, 0x7bc1e390, 0xa668711a, 0xac516b2e, 0x286e5a77, 0x5ab234a2, 0xbb742d8e, 0xe87d4f13, 0x13ceef81, 0x3b22ecc1, 0xfe3c3917, 0x6c6bf74f, 0x4fefad3a, 0xb122645f, 0xa273f277, 0xc59b1a79, 0x6d63b24e, 0xf0a3332, 0x972cd317, 0xc1cc8d5b, 0x143bf9a5, 0x47bd74bb, 0x999f033b, 0xde9447d6, 0x7d80cc07, 0xbe3f1a2a, 0x46bdf4ce, 0x82f95d8d, 0x33013fc4, 0x384db347, 0x102c24c4, 0xc1f82170, 0x8567a8fe, 0xae822f8b, 0x3d7c2fb1, 0xe27df801, 0xdc64d35a};
uint32_t power[] = {0x26aaea6d, 0x32bd5122, 0x88c9d5f9, 0x797542cf, 0x97d4959e, 0x459478aa, 0x622fff7a, 0x497cfe44, 0xa1b17c19, 0x25dbf182, 0x404d87e7, 0x9b8b869b, 0x42891e6c, 0x4745dc1, 0xab450a47, 0xb961a359, 0x4c6bc9f4, 0xfe79f939, 0xfd13779d, 0x88cca6ca, 0xc81bb5ef, 0xbc7f2f84, 0x4d0917d7, 0x35fe4ff7, 0x24545c43, 0x44c4a3ab, 0x5209083d, 0xfbba642e, 0xef4a751d, 0x4f3486bf, 0xffdedad, 0x85b857ca, 0xcffabfd7, 0x6669436d, 0xe29069dc, 0x77d114f9, 0x5ef6950e, 0xa21cc95f, 0xa78deb5b, 0x3a6e47b8, 0x398baf1e, 0x60404bc5, 0xcd53dfd4, 0xe4a62641, 0xba330a0, 0xe7fcacff, 0x2f7a67, 0x3f86b5d9, 0xb0de3253, 0x20f16120, 0x26c53e70, 0x491e4643, 0xe58af8cc, 0xf9ca62c2, 0x190b071c, 0xba044aa0, 0xac15fb39, 0xb4ca5a23, 0x69c8c4f7, 0xa257636c, 0x1c76a4e6, 0xe13df6a5, 0x986e9237, 0x862dc281};
uint32_t expo[] = {0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
uint32_t random_word_base_check(int x) {
  return basecheck[x];
}

void random_words_base_check(uint32_t *x, uint32_t count) {
  int index;

  for(index=count;index<count+32;index++)
    x[index-count]=random_word_base_check(index);
}
//power
uint32_t random_word_power(int x) {
  return power[x];
}
void random_words_power(uint32_t *x, uint32_t count) {
  int index;

  for(index=count;index<count+32;index++)
    x[index-count]=random_word_power(index);
}
//exp
uint32_t random_word_exp(int x) {
  return expo[x];
}

void random_words_exp(uint32_t *x, uint32_t count) {
  int index;

  for(index=count;index<count+32;index++)
    x[index-count]=random_word_exp(index);
}
template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x  
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
  
  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
};

template<class params>
class powm_odd_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable
  
  // define the instance structure
  typedef struct {
    cgbn_mem_t<params::BITS> x;
    cgbn_mem_t<params::BITS> power;
    cgbn_mem_t<params::BITS> modulus;
    cgbn_mem_t<params::BITS> result;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;
  
  __device__ __forceinline__ powm_odd_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
  }

  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);
    
    // convert x into Montgomery space, store into window table
    np0=cgbn_bn2mont(_env, result, x, modulus);
    cgbn_store(_env, window+1, result);
    cgbn_set(_env, t, result);
    
    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
      cgbn_store(_env, window+index, result);
    }

    // find leading high bit
    position=params::BITS - cgbn_clz(_env, power);

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, result, window+index);
    
    // process the remaining exponent chunks
    while(position>0) {
      // square the result window_bits times
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, result, result, modulus, np0);
      
      // multiply by next exponent chunk
      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
    }
    
    // we've processed the exponent now, convert back to normal space
    cgbn_mont2bn(_env, result, result, modulus, np0);
  }
  
  __device__ __forceinline__ void sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t         t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    bn_local_t   odd_powers[1<<window_bits-1];

    // conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
    // requires:  x<modulus,  modulus is odd
        
    // find the leading one in the power
    leading=params::BITS-1-cgbn_clz(_env, power);
    if(leading>=0) {
      // convert x into Montgomery space, store in the odd powers table
      mont_inv=cgbn_bn2mont(_env, result, x, modulus);
      
      // compute t=x^2 mod modulus
      cgbn_mont_sqr(_env, t, result, modulus, mont_inv);
      
      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn_store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn_store(_env, odd_powers+index, result);
      }
  
      // starts contains an array of bits indicating the start of a window
      cgbn_set_ui32(_env, starts, 0);
  
      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }
  
      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn_load(_env, result, odd_powers+index);
      position--;
      
      // Process remaining windows 
      while(position>=0) {
        cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) {
          // found a window, load the index
          index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn_load(_env, t, odd_powers+index);
          cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }
      
      // convert result from Montgomery space
      cgbn_mont2bn(_env, result, result, modulus, mont_inv);
    }
    else {
      // p=0, thus x^p mod modulus=1
      cgbn_set_ui32(_env, result, 1);
    }
  }
  
  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;
  
    for(index=0;index<count;index++) {
      random_words_base_check(instances[index].x._limbs,32*index);
      random_words_power(instances[index].power._limbs, 32*index);
      random_words_exp(instances[index].modulus._limbs, 32*index);

      // ensure modulus is odd
      instances[index].modulus._limbs[0] |= 1;

      // ensure modulus is greater than 
      if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)>0) {
       swap_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32);
        
        // modulus might now be even, ensure it's odd
       instances[index].modulus._limbs[0] |= 1;
      }
      else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)==0) {
        // since modulus is odd and modulus = x, we can just subtract 1 from x
       instances[index].x._limbs[0] -= 1;
     }
    }
    return instances;
  }
  
  __host__ static void verify_results(instance_t *instances, uint32_t count) {
    mpz_t x, p, m, computed;
    
    mpz_init(x);
    mpz_init(p);
    mpz_init(m);
    mpz_init(computed);
    
    for(int index=0;index<count;index++) {
      to_mpz(x, instances[index].x._limbs, params::BITS/32);
      to_mpz(p, instances[index].power._limbs, params::BITS/32);
      to_mpz(m, instances[index].modulus._limbs, params::BITS/32);
      to_mpz(computed, instances[index].result._limbs, params::BITS/32);
      if(mpz_cmp_ui(computed,1)== 0) {
         printf ("hi");
      }
      mpz_out_str (stdout, 16, x);
      printf ("\n");
      mpz_out_str (stdout, 16, p);
      printf ("\n");
      mpz_out_str (stdout, 16, m);
      printf ("\n");
      mpz_out_str (stdout, 16, computed);
      printf ("\n");
      }
    mpz_clear(x);
    mpz_clear(p);
    mpz_clear(m);
    mpz_clear(computed);
    
    printf("All results match\n");
  }
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the powm_odd_t class

template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;
  
  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(instances[instance].x));
  cgbn_load(po._env, p, &(instances[instance].power));
  cgbn_load(po._env, m, &(instances[instance].modulus));
  
  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.  
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);
  
  cgbn_store(po._env, &(instances[instance].result), r);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename powm_odd_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block
  
  printf("Genereating instances ...\n");
  instances=powm_odd_t<params>::generate_instances(instance_count);
  
  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));
  
  printf("Verifying the results ...\n");
  powm_odd_t<params>::verify_results(instances, instance_count);
  
  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  typedef powm_params_t<8, 1024, 5> params;
  run_test<params>(2);
  return 0;
}
