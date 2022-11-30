#include <pthread.h>
#include <unistd.h>
#define numthreads 4

struct matx{
	int N;
	int *matA;
	int *matB;
	int *output;
	int i;
};

void *mult(void* q) 
{
	int N;
	int *matA;
	int *matB;
	int *output;
	int l;

	struct matx *tp = (matx*)q;
	N=tp->N;
	matA=tp->matA;
	matB=tp->matB;
	output=tp->output;
	l=tp->i; 

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
	for (i = l; i < l+(N>>2); i+=2){
	    for (k = 0; k < N; k+=2){    
		sum=0;
		for (j = 0; j < N; j += 8){
		    
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
		    
		  
		    for(int m=0 ; m <8 ; m++) {   
		    	sum += arr[m];
		    	arr[m]=0;
		    }
		    
		}
		output[(i>>1)*(N>>1) + (k>>1)] = sum;
	    }	    	
	}
} 


void multiThread(int N, int *matA, int *matB, int *output)
{
    assert( N>=4 and N == ( N &~ (N-1)));
    
	    struct matx ti[numthreads];
	   
	    pthread_t tid[numthreads];
	    
	    for(int x=0;x<numthreads ;x++){
	       ti[x].N=N;
	       ti[x].matA=matA;
	       ti[x].matB=matB;
	       ti[x].output=output;
	       
	       int range=x*(N>>2);
	       
	       ti[x].i=range;
	       
	       pthread_create(&tid[x] , NULL , mult , &ti[x]);
	    }
	    
	    for(int y=0;y<numthreads ;y++){
	       pthread_join(tid[y] , NULL);
    	    }
}













