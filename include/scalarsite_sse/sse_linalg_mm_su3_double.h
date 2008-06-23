#ifndef QDP_SSE_LINALG_MM_SU3_DOUBLE_H
#define QDP_SSE_LINALG_MM_SU3_DOUBLE_H

#include "qdp_precision.h"
namespace QDP { 

  /* M = a*M  a is scalar */
  void ssed_m_eq_scal_m(REAL64* m2, REAL64* a, REAL *m1, int n_mat);

  /* M *= a,  a is a scalar */
  void ssed_m_muleq_scal(REAL64* m, REAL64* a, int n_mat);

  /* M2 += M1 */
  void ssed_m_peq_m(REAL64* m2, REAL64* m1, int n_mat);

  /* M2 -= M1 */
  void ssed_m_meq_m(REAL64* m2, REAL64* m1, int nmat);

  /* M2 += adj(M1) */
  void ssed_m_peq_h(REAL64* m2, REAL64* m1, int n_mat);

  /* M2 -= adj(M1) */
  void ssed_m_meq_h(REAL64* m2, REAL64* m1, int n_mat);

  /* M3 = M1*M2 */
  void ssed_m_eq_mm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 = M1*adj(M2) */
  void ssed_m_eq_mh(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 = adj(M1)*M2 */
  void ssed_m_eq_hm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 = adj(M1)*adj(M2) */
  void ssed_m_eq_mm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);

  /* M3 += M1*M2 */
  void ssed_m_peq_mm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 += M1*adj(M2) */
  void ssed_m_peq_mh(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 += adj(M1)*M2 */
  void ssed_m_peq_hm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 += adj(M1)*adj(M2) */
  void ssed_m_peq_hh(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);

  /* M3 -= M1*M2 */
  void ssed_m_meq_mm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 -= M1*adj(M2) */
  void ssed_m_meq_mh(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 -= adj(M1)*M2 */
  void ssed_m_meq_hm(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);
  
/* M3 -= adj(M1)*adj(M2) */
  void ssed_m_meq_hh(REAL64* m3, REAL64* m1, REAL64* m2, int n_mat);


  


}
#endif
