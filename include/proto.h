// -*- C++ -*-
//
// $Id: proto.h,v 1.8 2003-01-20 16:16:02 edwards Exp $
//
// QDP data parallel interface
//
// prototypes used throughout the QDP code

QDP_BEGIN_NAMESPACE(QDP);

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! Calculates the lexicographic site index from the coordinate of a site
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size);

//! Unique-ify a list
multi1d<int> uniquify_list(const multi1d<int>& ll);

//! Initializer for subsets
void InitDefaultSets();

//! Initializer for maps
void InitDefaultMaps();

//! Su2_extract: r_0,r_1,r_2,r_3 <- source(su2_index)  [SU(N) field]  under a subset
/*! 
 * @param source extract the su2 matrix from this su(n) gauge field
 * @param su2_index  int lying in [0, Nc*(Nc-1)/2)
 * @return su2 matrix represented in the O(4) rep. - an array of LatticeReal 
 * @ingroup group5
 * @relates QDPType
 *
 * Extract components r_k proportional to SU(2) submatrix su2_index
 * from the "SU(Nc)" matrix V. The SU(2) matrix is parametrized in the
 * sigma matrix basis.
 *
 * There are Nc*(Nc-1)/2 unique SU(2) submatrices in an SU(Nc) matrix.
 * The user does not need to know exactly which one is which, just that
 * they are unique.
 */
multi1d<LatticeReal> 
su2Extract(const LatticeColorMatrix& source, 
	   int su2_index, 
	   const Subset& s);


//! Sun_fill: dest(su2_index) <- r_0,r_1,r_2,r_3  under a subset
/*!
 * @param source su2 matrix represented in the O(4) rep. - an array of LatticeReal 
 * @param su2_index  int lying in [0, Nc*(Nc-1)/2)
 * @return su(n) matrix 
 * @ingroup group5
 * @relates QDPType
 *
 * Fill an SU(Nc) matrix V with the SU(2) submatrix su2_index
 * paramtrized by b_k in the sigma matrix basis.
 *
 * Fill in B from B_SU(2) = b0 + i sum_k bk sigma_k
 *
 * There are Nc*(Nc-1)/2 unique SU(2) submatrices in an SU(Nc) matrix.
 * The user does not need to know exactly which one is which, just that
 * they are unique.
 */
LatticeColorMatrix
sunFill(const multi1d<LatticeReal> r,
	int su2_index,
	const Subset& s);


//! fread on a binary file written in big-endian order
size_t bfread(void *ptr, size_t size, size_t nmemb, FILE *stream);

//! fwrite to a binary file in big-endian order
size_t bfwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);


//! Read a QCD (NERSC) Archive format gauge field
void readArchiv(multi1d<LatticeColorMatrix>& u, char file[]);

//! Read a SZIN format gauge field
void readSzin(multi1d<LatticeColorMatrix>& u, int cfg_io_location, char file[], Seed& seed_old);

//! Read a SZIN quark propagator
void readSzinQprop(LatticePropagator& q, char file[]);

QDP_END_NAMESPACE();

