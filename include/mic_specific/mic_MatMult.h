#ifdef __MIC

#ifndef INCL_mic_MatMult
#define INCL_mic_MatMult

#include "immintrin.h"

namespace QDP
{

typedef RComplex<REAL64>  RComplexD;
typedef PColorMatrix<RComplexD, 3> MatSU3;

inline void mic_MatMult(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
			//w=u*v;

			__m512d wvec, uvec;
			__m512d vrow1[3], vrow2[3];

// 			const __m512i indexv = _mm512_set_epi32(
//            0,0,0,0,
//            0,0,0,0,
//            0,0,4,4,
//            2,2,0,0);
//
// 			const __m512i indexu = _mm512_set_epi32(
//            0,0,0,0,
//            0,0,0,0,
//            0,0,1,0,
//            1,0,1,0);
//
// 			const __m512i indexw = _mm512_set_epi32(
//            0,0,0,0,
//            0,0,0,0,
//            0,0,5,4,
//            3,2,1,0);
//
//       const int scale=8;

      const __int64 sbit=((__int64)1<<63);
      const __m512i change_sign = _mm512_set_epi64(0,0,0,sbit,0,sbit,0,sbit);

      for (int k=0; k<3; ++k)
			{

					// a triple broadcast is faster than gather which is faster than set
					//vrow1[k] = _mm512_set_pd(0, 0, v.elem(k,2).real(), v.elem(k,2).real(), v.elem(k,1).real(), v.elem(k,1).real(), v.elem(k,0).real(), v.elem(k,0).real());
					//vrow2[k] = _mm512_set_pd(0, 0, v.elem(k,2).imag(), -v.elem(k,2).imag(), v.elem(k,1).imag(), -v.elem(k,1).imag(), v.elem(k,0).imag(), -v.elem(k,0).imag());
// 					vrow1[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).real(), scale);

					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(k,0).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(k,1).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(k,2).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

					// fill with imaginary parts using the gather pattern and change signs with xor
					// seems to work correctly but beware with the undocumented sign bit
// 					vrow2[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).imag(), scale);
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(k,0).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(k,1).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(k,2).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

			}

			// compute w row after row
			for(int i=0; i < 3; ++i)
			{
				for (int k=0; k<3; ++k)
				{
					// a double broadcast is faster than gather which is faster than set
					//uvec = _mm512_set_pd(0, 0, u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real());
					//uvec = _mm512_i32logather_pd(indexu, (void*)&u.elem(i,k).real(), scale);

					// broadcast real part
					uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(i,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					// broadcast imaginary part
					uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(i,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

					if (!k)
						//wvec = _mm512_mul_pd(uvec,vrow1[k]);
						wvec = _mm512_mask_mul_pd(wvec,0x3F,uvec,vrow1[k]);
					else
						wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
						//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
						// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

					uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

					wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
					//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

				}
				_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
				_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

				//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
			}
}

inline void mic_MatMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=u*adj(v);		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!

			__m512d wvec, uvec;
			__m512d vrow1[3], vrow2[3];

      const __int64 sbit=((__int64)1<<63);
      const __m512i change_sign = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);

      for (int k=0; k<3; ++k)
			{
					// a triple broadcast is faster than gather which is faster than set
//  					vrow1[k] = _mm512_set_pd(0, 0, v.elem(2,k).real(), v.elem(2,k).real(), v.elem(1,k).real(), v.elem(1,k).real(), v.elem(0,k).real(), v.elem(0,k).real());
// 					vrow2[k] = _mm512_set_pd(0, 0, -v.elem(2,k).imag(), v.elem(2,k).imag(), -v.elem(1,k).imag(), v.elem(1,k).imag(), -v.elem(0,k).imag(), v.elem(0,k).imag());

					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(0,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(1,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(2,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

					// fill with imaginary parts using the gather pattern and change signs with xor
					// seems to work correctly but beware with the undocumented sign bit
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(0,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(1,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(2,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

			}

			// compute w row after row
			for(int i=0; i < 3; ++i)
			{
				for (int k=0; k<3; ++k)
				{
					// a double broadcast is faster than gather which is faster than set
					//uvec = _mm512_set_pd(0, 0, u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real());
					//uvec = _mm512_i32logather_pd(indexu, (void*)&u.elem(i,k).real(), scale);

					// broadcast real part
					uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(i,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
					// broadcast imaginary part
					uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(i,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

					if (!k)
						//wvec = _mm512_mul_pd(uvec,vrow1[k]);
						wvec = _mm512_mask_mul_pd(wvec,0x3F,uvec,vrow1[k]);
					else
						wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
						//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
						// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

					uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

					wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
					//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

				}
				_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
				_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);
			}
}

inline void mic_MatAdjMult(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=adj(u)*v;		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!
//	return;

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,0,sbit,0,sbit,0,sbit);
	const __m512i change_signu = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);

	for (int k=0; k<3; ++k)
	{

			// a triple broadcast is faster than gather which is faster than set
			//vrow1[k] = _mm512_set_pd(0, 0, v.elem(k,2).real(), v.elem(k,2).real(), v.elem(k,1).real(), v.elem(k,1).real(), v.elem(k,0).real(), v.elem(k,0).real());
			//vrow2[k] = _mm512_set_pd(0, 0, v.elem(k,2).imag(), -v.elem(k,2).imag(), v.elem(k,1).imag(), -v.elem(k,1).imag(), v.elem(k,0).imag(), -v.elem(k,0).imag());
// 					vrow1[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).real(), scale);

			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(k,0).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(k,1).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(k,2).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

			// fill with imaginary parts using the gather pattern and change signs with xor
			// seems to work correctly but beware with the undocumented sign bit
// 					vrow2[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).imag(), scale);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(k,0).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(k,1).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(k,2).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real());

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(k,i).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(k,i).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// change sign of imaginary parts
			uvec = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(uvec),change_signu));

			if (!k)
				//wvec = _mm512_mul_pd(uvec,vrow1[k]);
				wvec = _mm512_mask_mul_pd(wvec,0x3F,uvec,vrow1[k]);
			else
				wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
				//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

		//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
	}
}

inline void mic_MatAdjMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=adj(u)*adj(v);		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!
//	return;

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);	// same for u and v

	for (int k=0; k<3; ++k)
	{
		// a triple broadcast is faster than gather which is faster than set
//  				vrow1[k] = _mm512_set_pd(0, 0, v.elem(2,k).real(), v.elem(2,k).real(), v.elem(1,k).real(), v.elem(1,k).real(), v.elem(0,k).real(), v.elem(0,k).real());
// 					vrow2[k] = _mm512_set_pd(0, 0, -v.elem(2,k).imag(), v.elem(2,k).imag(), -v.elem(1,k).imag(), v.elem(1,k).imag(), -v.elem(0,k).imag(), v.elem(0,k).imag());

		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(0,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(1,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(2,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

		// fill with imaginary parts using the gather pattern and change signs with xor
		// seems to work correctly but beware with the undocumented sign bit
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(0,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(1,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(2,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real());

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(k,i).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(k,i).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// change sign of imaginary parts
			uvec = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(uvec),change_sign));

			if (!k)
				//wvec = _mm512_mul_pd(uvec,vrow1[k]);
				wvec = _mm512_mask_mul_pd(wvec,0x3F,uvec,vrow1[k]);
			else
				wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
				//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

		//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
	}
}

//*********************************************************************************************
inline void mic_AddMatMult(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
	//w=u*v;
	//return;

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,0,sbit,0,sbit,0,sbit);

	for (int k=0; k<3; ++k)
	{
		// a triple broadcast is faster than gather which is faster than set
		//vrow1[k] = _mm512_set_pd(0, 0, v.elem(k,2).real(), v.elem(k,2).real(), v.elem(k,1).real(), v.elem(k,1).real(), v.elem(k,0).real(), v.elem(k,0).real());
		//vrow2[k] = _mm512_set_pd(0, 0, v.elem(k,2).imag(), -v.elem(k,2).imag(), v.elem(k,1).imag(), -v.elem(k,1).imag(), v.elem(k,0).imag(), -v.elem(k,0).imag());
// 					vrow1[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).real(), scale);

		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(k,0).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(k,1).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(k,2).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

		// fill with imaginary parts using the gather pattern and change signs with xor
		// seems to work correctly but beware with the undocumented sign bit
// 					vrow2[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).imag(), scale);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(k,0).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(k,1).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(k,2).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		// load w
		wvec = _mm512_mask_loadunpacklo_pd(wvec, 0x3F,(void*)&w.elem(i,0).real());
		wvec = _mm512_mask_loadunpackhi_pd(wvec, 0x3F,(void*)&w.elem(i,0).real()+64);

		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real());
			//uvec = _mm512_i32logather_pd(indexu, (void*)&u.elem(i,k).real(), scale);

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(i,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(i,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

			wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

		//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
	}
}

inline void mic_AddMatMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=u*adj(v);		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);

	for (int k=0; k<3; ++k)
	{
		// a triple broadcast is faster than gather which is faster than set
//  				vrow1[k] = _mm512_set_pd(0, 0, v.elem(2,k).real(), v.elem(2,k).real(), v.elem(1,k).real(), v.elem(1,k).real(), v.elem(0,k).real(), v.elem(0,k).real());
// 					vrow2[k] = _mm512_set_pd(0, 0, -v.elem(2,k).imag(), v.elem(2,k).imag(), -v.elem(1,k).imag(), v.elem(1,k).imag(), -v.elem(0,k).imag(), v.elem(0,k).imag());

		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(0,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(1,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(2,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

		// fill with imaginary parts using the gather pattern and change signs with xor
		// seems to work correctly but beware with the undocumented sign bit
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(0,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(1,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(2,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		// load w
		wvec = _mm512_mask_loadunpacklo_pd(wvec, 0x3F,(void*)&w.elem(i,0).real());
		wvec = _mm512_mask_loadunpackhi_pd(wvec, 0x3F,(void*)&w.elem(i,0).real()+64);

		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real());
			//uvec = _mm512_i32logather_pd(indexu, (void*)&u.elem(i,k).real(), scale);

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(i,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(i,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

			wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);
	}
}

inline void mic_AddMatAdjMult(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=adj(u)*v;		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!
//	return;

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,0,sbit,0,sbit,0,sbit);
	const __m512i change_signu = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);

	for (int k=0; k<3; ++k)
	{

			// a triple broadcast is faster than gather which is faster than set
			//vrow1[k] = _mm512_set_pd(0, 0, v.elem(k,2).real(), v.elem(k,2).real(), v.elem(k,1).real(), v.elem(k,1).real(), v.elem(k,0).real(), v.elem(k,0).real());
			//vrow2[k] = _mm512_set_pd(0, 0, v.elem(k,2).imag(), -v.elem(k,2).imag(), v.elem(k,1).imag(), -v.elem(k,1).imag(), v.elem(k,0).imag(), -v.elem(k,0).imag());
// 					vrow1[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).real(), scale);

			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(k,0).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(k,1).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(k,2).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

			// fill with imaginary parts using the gather pattern and change signs with xor
			// seems to work correctly but beware with the undocumented sign bit
// 					vrow2[k] = _mm512_i32logather_pd(indexv, (void*)&v.elem(k,0).imag(), scale);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(k,0).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(k,1).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(k,2).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		// load w
		wvec = _mm512_mask_loadunpacklo_pd(wvec, 0x3F,(void*)&w.elem(i,0).real());
		wvec = _mm512_mask_loadunpackhi_pd(wvec, 0x3F,(void*)&w.elem(i,0).real()+64);

		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real());

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(k,i).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(k,i).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// change sign of imaginary parts
			uvec = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(uvec),change_signu));

			wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

		//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
	}
}

inline void mic_AddMatAdjMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v)
{
//	w=adj(u)*adj(v);		// absolutely absurd: for single SU3 mult it calls adjoint and the SU3 multiplication!!!
//	return;

	__m512d wvec, uvec;
	__m512d vrow1[3], vrow2[3];

	const __int64 sbit=((__int64)1<<63);
	const __m512i change_sign = _mm512_set_epi64(0,0,sbit,0,sbit,0,sbit,0);	// same for u and v

	for (int k=0; k<3; ++k)
	{
		// a triple broadcast is faster than gather which is faster than set
//  				vrow1[k] = _mm512_set_pd(0, 0, v.elem(2,k).real(), v.elem(2,k).real(), v.elem(1,k).real(), v.elem(1,k).real(), v.elem(0,k).real(), v.elem(0,k).real());
// 					vrow2[k] = _mm512_set_pd(0, 0, -v.elem(2,k).imag(), v.elem(2,k).imag(), -v.elem(1,k).imag(), v.elem(1,k).imag(), -v.elem(0,k).imag(), v.elem(0,k).imag());

		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x03, (void*)&v.elem(0,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x0C, (void*)&v.elem(1,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow1[k] = _mm512_mask_extload_pd(vrow1[k], 0x30, (void*)&v.elem(2,k).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);

		// fill with imaginary parts using the gather pattern and change signs with xor
		// seems to work correctly but beware with the undocumented sign bit
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x03, (void*)&v.elem(0,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x0C, (void*)&v.elem(1,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_mask_extload_pd(vrow2[k], 0x30, (void*)&v.elem(2,k).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
		vrow2[k] = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vrow2[k]),change_sign));

	}

	// compute w row after row
	for(int i=0; i < 3; ++i)
	{
		// load w
		wvec = _mm512_mask_loadunpacklo_pd(wvec, 0x3F,(void*)&w.elem(i,0).real());
		wvec = _mm512_mask_loadunpackhi_pd(wvec, 0x3F,(void*)&w.elem(i,0).real()+64);

		for (int k=0; k<3; ++k)
		{
			// a double broadcast is faster than gather which is faster than set
			//uvec = _mm512_set_pd(0, 0, -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real(), -u.elem(k,i).imag(), u.elem(k,i).real());

			// broadcast real part
			uvec = _mm512_mask_extload_pd(uvec, 0x15, (void*)&u.elem(k,i).real(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// broadcast imaginary part
			uvec = _mm512_mask_extload_pd(uvec, 0x2A, (void*)&u.elem(k,i).imag(), _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
			// change sign of imaginary parts
			uvec = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(uvec),change_sign));

			wvec = _mm512_fmadd_pd(uvec,vrow1[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow1[k],wvec,0x3F);

//-------------------
				// swap hgfe dcba to  ghef cdab
//					__declspec(align(64)) double uu[8]={u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(), u.elem(i,k).imag(), u.elem(i,k).real(),0,0};

			uvec = __cdecl _mm512_swizzle_pd(uvec, _MM_SWIZ_REG_CDAB);

			wvec = _mm512_fmadd_pd(uvec,vrow2[k],wvec);
			//wvec = _mm512_mask3_fmadd_pd(uvec,vrow2[k],wvec,0x3F);

//-------------------

		}
		_mm512_mask_packstorelo_pd((void*)&w.elem(i,0).real(), 0x3F, wvec);
		_mm512_mask_packstorehi_pd((void*)&w.elem(i,0).real()+64, 0x3F, wvec);

		//_mm512_mask_i32loscatter_pd((void*)&w.elem(i,0).real(), 0x3F, indexw, wvec, scale);
	}
}

}	// namespace QDP

#endif

#endif // __MIC