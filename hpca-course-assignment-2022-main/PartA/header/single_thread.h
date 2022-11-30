#include <immintrin.h>

void singleThread(int N, int *matA, int *matB, int *output)  
{
  	int *arr = new int[N];
  	int *matT = new int[N*N];
  	
  	for (int r=0 ; r < N ; r++){
  	     for (int c=0 ; c < N ; c++)
  		  matT[r*N + c] = matB[c*N + r] ;
  	}
  	
  	
	__m256i matC = _mm256_setzero_si256();
	__m256i vec1 = _mm256_setzero_si256();
	__m256i vec2 = _mm256_setzero_si256();
	__m256i vec3 = _mm256_setzero_si256();
	__m256i vec4 = _mm256_setzero_si256();
	__m256i vec5 = _mm256_setzero_si256();
	__m256i vec6 = _mm256_setzero_si256();
	__m256i vec7 = _mm256_setzero_si256();

	int i, j, k, sum=0;
	for (i = 0; i < N; i+=2){
	    for (k = 0; k < N; k+=2){
	        
		sum=0;
		
		for (j = 0; j < N; j += 8)
		{
		    int x = k*N + j;
		    int y = i*N + j;
		     
		    vec1 = _mm256_loadu_si256((__m256i*)&matA[y]); 
		    vec2 = _mm256_loadu_si256((__m256i*)&matA[y+N]); 
		    vec3 = _mm256_loadu_si256((__m256i*)&matT[x]);
		    vec4 = _mm256_loadu_si256((__m256i*)&matT[x+N]);
		    
		    
		    vec5 = _mm256_add_epi32(vec1, vec2);
		    vec6 = _mm256_add_epi32(vec3, vec4);
		    
		    vec7 = _mm256_mullo_epi32(vec5, vec6);
		    
		    _mm256_storeu_si256((__m256i*) &arr[0], vec7);
		    
		  
		    for(int l=0 ; l <8 ; l++){   
		    	sum += arr[l];
		    	arr[l]=0;
		    }
		}
		output[(i>>1)*(N>>1) + (k>>1)] = sum;
	    }
	}
}  
