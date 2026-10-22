/* Independent C wire fixtures with fixed offsets and known numeric values. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {uint64_t n;int mode,columns,pairs;} OriginalVectorHeader;
_Static_assert(sizeof(OriginalVectorHeader)==24,"C vector wire layout changed");
size_t openmx_test_vector_packet_golden(void *destination,int mode){
    OriginalVectorHeader header;memset(&header,0,sizeof(header));
    header.n=3;header.mode=mode;header.columns=3;header.pairs=4;
    const double data[9]={2,3,4,0,2,-2,-3,1,2};
    const double weights[3]={2,-1,.5},coefficients[3]={2,-.5,1};
    const int left[4]={2,0,1,2},right[4]={0,1,1,2};
    const size_t bytes=mode?120:152;
    if(destination){
        unsigned char *p=destination;memcpy(p,&header,24);memcpy(p+24,data,72);
        memcpy(p+96,mode?coefficients:weights,24);
        if(!mode){memcpy(p+120,left,16);memcpy(p+136,right,16);}
    }
    return bytes;
}
void openmx_test_vector_packet_free(void *pointer){free(pointer);}
