/* Independent C wire fixture: one fixed slice of dense/CSR/lookup data.
   Offsets are fixture constants, not calls to the production layout code. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t first,count,parameter_bytes;
    int inputs,outputs,scratch,scope_kind;
    struct { uint64_t bytes,item_bytes,element_bytes; } view[40];
} OriginalTaskHeader;
_Static_assert(sizeof(OriginalTaskHeader)==1000,"C wire header changed");

size_t openmx_test_task_packet_golden(void *destination)
{
    OriginalTaskHeader header={0};
    const unsigned char parameters[13]={2,3,4,5,6,7,8,9,10,11,12,13,14};
    const float dense[4]={.5f,3.f,-2.f,7.f};
    const size_t offsets[5]={0,1,3,3,6};
    const int items[6]={4,-3,6,8,-4,5},lookup[2]={5,-1};
    /* Both native size_t widths retain 16-byte section alignment. */
    const size_t index_section=sizeof(size_t)==8?48:32;
    const size_t lookup_offset=1040+index_section+32,bytes=lookup_offset+16;
    _Static_assert(sizeof(size_t)==4 || sizeof(size_t)==8,"unsupported index width");
    header.first=13;header.count=4;header.parameter_bytes=13;
    header.inputs=3;header.outputs=2;header.scratch=1;header.scope_kind=2;
    header.view[0].item_bytes=sizeof(float);
    header.view[1].element_bytes=sizeof(int);
    header.view[2].bytes=sizeof(lookup);
    header.view[3].item_bytes=sizeof(double);
    header.view[4].bytes=2*sizeof(double);
    header.view[5].item_bytes=sizeof(double);
    if(destination){
        unsigned char *p=destination;
        memset(p,0,bytes);memcpy(p,&header,sizeof(header));
        memcpy(p+1008,parameters,sizeof(parameters));memcpy(p+1024,dense,sizeof(dense));
        memcpy(p+1040,offsets,sizeof(offsets));memcpy(p+1040+index_section,items,sizeof(items));
        memcpy(p+lookup_offset,lookup,sizeof(lookup));
    }
    return bytes;
}

void openmx_test_task_packet_free(void *pointer){free(pointer);}
