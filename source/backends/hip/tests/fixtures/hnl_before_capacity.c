/* CASE 1 */
#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(items, krowA, krowB, krow_halo, krow_ene, krow_nlp, cdm_pref, halo, ene, item_f, flat)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const int k_off = items[pp].k_off;
                const int k_n = items[pp].k_n;
                const size_t cdm_off = items[pp].cdm_off;
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;


                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_pref[cdm_off + (size_t)idx];
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    for (int kk = 0; kk < k_n; kk++) {
                        const int nlp = krow_nlp[k_off + kk];
                        const size_t abase = krowA[k_off + kk] + (size_t)m * (size_t)nlp;
                        const double* b0 = (krow_halo[k_off + kk]
                            ? (halo + krowB[k_off + kk])
                            : (flat + krowB[k_off + kk])) + (size_t)n * (size_t)nlp;
                        const double* a1 = flat + flat_stride + abase;
                        const double* a2 = flat + 2U * flat_stride + abase;
                        const double* a3 = flat + 3U * flat_stride + abase;
                        const double* el = ene + krow_ene[k_off + kk];
                        double tx = 0.0, ty = 0.0, tz = 0.0;

                        for (int l = 0; l < nlp; l++) {
                            tx += el[l] * a1[l] * b0[l];
                            ty += el[l] * a2[l] * b0[l];
                            tz += el[l] * a3[l] * b0[l];
                        }
                        sx += tx;
                        sy += ty;
                        sz += tz;
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }

/* CASE 2 */
#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(items, item_h0, item_nlp, item_ene, item_a, item_b, cdm_pref, c2, ene, item_f, flat)
            for (int pp = 0; pp < nitems_c; pp++) {
                const int ian = items[pp].ian;
                const int jan = items[pp].jan;
                const size_t cdm_off = items[pp].cdm_off;
                const int h0 = item_h0[pp];
                const int nlp = item_nlp[pp];
                const size_t a_off = item_a[pp];
                const size_t b_off = item_b[pp];
                const double* el = ene + item_ene[pp];
                const int mn = ian * jan;
                double fx = 0.0, fy = 0.0, fz = 0.0;


                for (int idx = 0; idx < mn; idx++) {
                    const int m = idx / jan;
                    const int n = idx - m * jan;
                    const double w = cdm_pref[cdm_off + (size_t)idx];
                    double sx = 0.0, sy = 0.0, sz = 0.0;

                    if (h0 == 1) {
                        /* -sum_l ene * <i|k>(0) * d<j|k>(dir) */
                        const double* a0 = flat + a_off + (size_t)m * (size_t)nlp;
                        const double* b1 = c2 + c2_stride + b_off + (size_t)n * (size_t)nlp;
                        const double* b2 = c2 + 2U * c2_stride + b_off + (size_t)n * (size_t)nlp;
                        const double* b3 = c2 + 3U * c2_stride + b_off + (size_t)n * (size_t)nlp;

                        for (int l = 0; l < nlp; l++) {
                            sx -= el[l] * a0[l] * b1[l];
                            sy -= el[l] * a0[l] * b2[l];
                            sz -= el[l] * a0[l] * b3[l];
                        }
                    } else {
                        /* -sum_l ene * d<i|k>(dir) * <j|k>(0) */
                        const double* a1 = c2 + c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* a2 = c2 + 2U * c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* a3 = c2 + 3U * c2_stride + a_off + (size_t)m * (size_t)nlp;
                        const double* b0 = c2 + b_off + (size_t)n * (size_t)nlp;

                        for (int l = 0; l < nlp; l++) {
                            sx -= el[l] * a1[l] * b0[l];
                            sy -= el[l] * a2[l] * b0[l];
                            sz -= el[l] * a3[l] * b0[l];
                        }

                        if (h0 == 0) {
                            /* -sum_l ene * <i|k>(0) * d<j|k>(dir) */
                            const double* a0 = c2 + a_off + (size_t)m * (size_t)nlp;
                            const double* b1 = c2 + c2_stride + b_off + (size_t)n * (size_t)nlp;
                            const double* b2 = c2 + 2U * c2_stride + b_off + (size_t)n * (size_t)nlp;
                            const double* b3 = c2 + 3U * c2_stride + b_off + (size_t)n * (size_t)nlp;

                            for (int l = 0; l < nlp; l++) {
                                sx -= el[l] * a0[l] * b1[l];
                                sy -= el[l] * a0[l] * b2[l];
                                sz -= el[l] * a0[l] * b3[l];
                            }
                        }
                    }

                    fx += w * sx;
                    fy += w * sy;
                    fz += w * sz;
                }

                item_f[3 * (size_t)pp + 0] = fx;
                item_f[3 * (size_t)pp + 1] = fy;
                item_f[3 * (size_t)pp + 2] = fz;
            }
