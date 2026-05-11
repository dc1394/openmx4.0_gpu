#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Number of spatial dimensions (x, y, z)
#define DIM 3

/**
 * Recursively generate non-decreasing combinations of indices of length p.
 *xo - p: tensor order
 * - start: minimum index to allow
 * - depth: current recursion depth
 * - current: array to store the current combination
 * - combo_list: pointer to array of saved combinations
 * - combo_count: pointer to current number of combinations
 */
void generate_combinations(int p, int start, int depth,
                           int *current,
                           int ***combo_list, int *combo_count) 
{
  int i;
  if (depth == p) {
    *combo_list = realloc(*combo_list,
			  sizeof(int*) * (*combo_count + 1));
    (*combo_list)[*combo_count] = malloc(sizeof(int) * p);
    memcpy((*combo_list)[*combo_count], current,
	   sizeof(int) * p);
    (*combo_count)++;
    return;
  }
  for (i = start; i < DIM; ++i) {
    current[depth] = i;
    generate_combinations(p, i, depth + 1,
			  current, combo_list, combo_count);
  }
}

/**
 * Initialize independent tensor components for order p.
 * - out_list: will point to array of index arrays
 * - out_count: will be set to number of independent combinations
 */
void init_independent_components(int p, int ***out_list, int *out_count)
{
  int *buffer;
  *out_count = 0;
  *out_list = NULL;
  buffer = malloc(sizeof(int) * p);
  generate_combinations(p, 0, 0, buffer, out_list, out_count);
  free(buffer);
}

/**
 * Compare two index arrays of length p for equality.
 * Returns 1 if equal, 0 otherwise.
 */
int equal_indices(const int *a, const int *b, int p) {
  int i;
  for (i = 0; i < p; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

/**
 * Map a symmetric index array of length p to its index in the independent list.
 * - idx: array of length p containing indices (e.g., {0,1} for x,y)
 * - ind_list: list of independent combinations
 * - ind_count: number of entries in ind_list
 * Returns the matching index or -1 if not found.
 */


int map_to_independent(int p, const int *idx,
                       int **ind_list, int ind_count) {
  int *sorted = malloc(sizeof(int) * p);
  int i, j, result = -1;
  // Copy and sort
  for (i = 0; i < p; ++i) sorted[i] = idx[i];
  for (i = 0; i < p - 1; ++i) {
    for (j = i + 1; j < p; ++j) {
      if (sorted[i] > sorted[j]) {
	int tmp = sorted[i];
	sorted[i] = sorted[j];
	sorted[j] = tmp;
      }
    }
  }
  // Find in list
  for (i = 0; i < ind_count; ++i) {
    if (equal_indices(sorted, ind_list[i], p)) {
      result = i;
      break;
    }
  }
  free(sorted);
  return result;
}




int main(void) {
  int p = 3;  // Tensor order (change as needed)
  int **independent_list;
  int independent_count;
  int i;
    
  // Example: dynamic full index array of length p
  int *full_idx = malloc(sizeof(int) * p);
  // Fill full_idx with desired indices, e.g., y,z for p=2
  full_idx[0] = 2; 
  full_idx[1] = 1; 
  full_idx[2] = 2; 
  full_idx[3] = 0; 

  // Build independent list
  init_independent_components(p,
			      &independent_list,
			      &independent_count);

  // Print independent components
  printf("Independent components for p=%d (total %d):\n", p, independent_count);
  for (i = 0; i < independent_count; ++i) {
    int j;
    printf("[%2d] (", i);
    for (j = 0; j < p; ++j) {

      //printf("%c", 'x' + independent_list[i][j]);
      printf("%d",independent_list[i][j]);

      if (j < p - 1) printf(",");
    }
    printf(")\n");
  }

  // Map example using dynamic full_idx
  int map_index = map_to_independent(p, full_idx,
				     independent_list,
				     independent_count);
  printf("full_idx { ");
  for (i = 0; i < p; ++i) {
    printf("%c", 'x' + full_idx[i]);
    if (i < p - 1) printf(", ");
  }
  printf(" } maps to independent index [%d]\n", map_index);

  // Cleanup
  free(full_idx);
  for (i = 0; i < independent_count; ++i) {
    free(independent_list[i]);
  }
  free(independent_list);
  return 0;
}
