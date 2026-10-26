#ifndef OPENMX_CORE_FORCE_VNA_SIMD_H
#define OPENMX_CORE_FORCE_VNA_SIMD_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The projector-block part of the force family (openmx_core_force_simd.h),
   self-contained so that extracted Force4B code compiles on its own. Same
   contract as the rest of the family: mode is an OPENMX_GRID_* host SIMD
   mode, every entry returns 0 and writes nothing when it refuses a request,
   no allocation, caller-owned thread-private scratch. */

#define OPENMX_FORCE_SIMD_PAD 16

/* A lane operand prepared once and shared by several block calls: the rows of
   each component transposed to a plane [projector][pad] of floats, pad = count
   rounded up to OPENMX_FORCE_SIMD_PAD, planes consecutive per component.
   openmx_core_force_vna_lanes fills caller memory of at least
   openmx_core_force_vna_lanes_bytes() bytes from rows[c][n] (row n of
   component c, projectors floats each) and describes it in *lanes. */
typedef struct {
  const float *planes;
  int count, projectors, components;
  size_t pad;
} OpenMXForceVnaLanes;
size_t openmx_core_force_vna_lanes_bytes(int count,int projectors,int components);
int openmx_core_force_vna_lanes(const float *const *const *rows,int components,int count,int projectors,
    void *memory,size_t bytes,OpenMXForceVnaLanes *lanes,int mode);

/* Neutral-atom projector derivative block of the fused Force4B traces:
     terms 1:  H_c[m][n] =   sum_l  derivative[c][m][l] * value[n][l]
     terms 2:  H_c[m][n] = - sum_l (derivative[c][m][l] * value[n][l]
                                   + row_value[m][l] * column_derivative[c][n][l])
   for c = x,y,z (the two products of a term-2 step are added first, then
   subtracted, as in Force4B_case2_trace_fused). fp32 != 0 rounds each float
   product in float before widening, else both operands widen first. Rows are
   float projector rows of at least projectors entries.
   lanes_rows selects the operand the SIMD lanes span: 0 = the columns n (the
   value side: planes value, column_derivative x,y,z), 1 = the rows m (the
   derivative side: planes derivative x,y,z, row_value); the other operand is
   streamed row by row. The lane operand is read from *lanes when given (count
   = that side's extent, same projectors, enough components), else transposed
   into scratch of openmx_core_force_vna_scratch(lane count, projectors, terms,
   lanes_rows) bytes. output[c] receives H_c at [m*output_stride + n] for
   lanes_rows 0 and at [n*output_stride + m] for lanes_rows 1 (the stride is
   at least the lane count). */
typedef struct {
  int rows, columns, projectors, terms, fp32, lanes_rows;
  const float *const *derivative[3];        /* [rows] */
  const float *const *row_value;            /* [rows], terms 2 */
  const float *const *value;                /* [columns] */
  const float *const *column_derivative[3]; /* [columns], terms 2 */
  const OpenMXForceVnaLanes *lanes;         /* optional prepared lane operand */
  double *output[3];
  size_t output_stride;
} OpenMXForceVnaBlock;
size_t openmx_core_force_vna_scratch(int lane_count,int projectors,int terms,int lanes_rows);
int openmx_core_force_vna_block(const OpenMXForceVnaBlock*,void *scratch,size_t scratch_bytes,int mode);

#ifdef __cplusplus
}
#endif
#endif
