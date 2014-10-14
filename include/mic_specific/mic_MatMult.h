#ifdef __MIC

#ifndef INCL_mic_MatMult
#define INCL_mic_MatMult

namespace QDP
{

	extern int debug_mic_MatMult;

	typedef RComplex<REAL64>  RComplexD;
	typedef PColorMatrix<RComplexD, 3> MatSU3;

	void mic_MatMult(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_MatMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_MatAdjMult(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_MatAdjMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_AddMatMult(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_SubMatMult(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_AddMatMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_SubMatMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_AddMatAdjMult(MatSU3& w, const MatSU3& u, const MatSU3& v);
	void mic_AddMatAdjMultAdj(MatSU3& w, const MatSU3& u, const MatSU3& v);

	// u *= v
	inline MatSU3 &operator*=(MatSU3 &u, const MatSU3 &v)
	{
		MatSU3 tmp=u;
		//u = tmp*v;
		mic_MatMult(u,tmp,v);
		return u;
	}


	// specializations of double precision colormatrix product for MIC

#ifdef WRONGTYPE
	// PMatrix = PMatrix * PMatrix
	template<>
	inline BinaryReturn<MatSU3, MatSU3, OpMultiply>::Type_t
	operator*(const MatSU3& l, const MatSU3& r)
	{
		BinaryReturn<MatSU3, MatSU3, OpMultiply>::Type_t  d;
	//QDPIO::cout << "W=U*V" << endl;

		mic_MatMult(d, l, r);

		return d;
	}
	// Optimized  PMatrix = adj(PMatrix)*PMatrix
	template<>
	inline BinaryReturn<MatSU3, MatSU3, OpAdjMultiply>::Type_t
	adjMultiply(const MatSU3& l, const MatSU3& r)
	{
		BinaryReturn<MatSU3, MatSU3, OpAdjMultiply>::Type_t  d;
	//QDPIO::cout << "W=adj(U)*V" << endl;

		mic_MatAdjMult(d, l, r);

		return d;
	}
	// Optimized  PMatrix = PMatrix*adj(PMatrix)
	template<>
	inline BinaryReturn<MatSU3, MatSU3, OpMultiplyAdj>::Type_t
	multiplyAdj(const MatSU3& l, const MatSU3& r)
	{
		BinaryReturn<MatSU3, MatSU3, OpMultiplyAdj>::Type_t  d;
	//QDPIO::cout << "W=U*adj(V)" << endl;

		mic_MatMultAdj(d, l, r);

		return d;
	}
	// Optimized  PMatrix = adj(PMatrix)*adj(PMatrix)
	template<>
	inline BinaryReturn<MatSU3, MatSU3, OpAdjMultiplyAdj>::Type_t
	adjMultiplyAdj(const MatSU3& l, const MatSU3& r)
	{
		BinaryReturn<MatSU3, MatSU3, OpAdjMultiplyAdj>::Type_t  d;
	//QDPIO::cout << "W=adj(U)*adj(V)" << endl;

		mic_MatAdjMultAdj(d, l, r);

		return d;
	}
#endif

	typedef PScalar<MatSU3> ScalMatSU3;

	template<>
	inline BinaryReturn<ScalMatSU3, ScalMatSU3, OpMultiply>::Type_t
	operator*(const ScalMatSU3& l, const ScalMatSU3& r)
	{
		BinaryReturn<ScalMatSU3, ScalMatSU3, OpMultiply>::Type_t  d;
	//QDPIO::cout << "ScaW=ScaU*ScaV" << endl;

		mic_MatMult(d.elem(), l.elem(), r.elem());

		return d;
	}

	template<>
	inline BinaryReturn<ScalMatSU3, ScalMatSU3, OpAdjMultiply>::Type_t
	adjMultiply(const ScalMatSU3& l, const ScalMatSU3& r)
	{
		BinaryReturn<ScalMatSU3, ScalMatSU3, OpAdjMultiply>::Type_t  d;
	//QDPIO::cout << "ScaW=adj(ScaU)*ScaV" << endl;

		mic_MatAdjMult(d.elem(), l.elem(), r.elem());

		return d;
	}

	template<>
	inline BinaryReturn<ScalMatSU3, ScalMatSU3, OpMultiplyAdj>::Type_t
	multiplyAdj(const ScalMatSU3& l, const ScalMatSU3& r)
	{
		BinaryReturn<ScalMatSU3, ScalMatSU3, OpMultiplyAdj>::Type_t  d;
	//QDPIO::cout << "ScaW=ScaU*adj(ScaV)" << endl;

		mic_MatMultAdj(d.elem(), l.elem(), r.elem());

		return d;
	}

	template<>
	inline BinaryReturn<ScalMatSU3, ScalMatSU3, OpAdjMultiplyAdj>::Type_t
	adjMultiplyAdj(const ScalMatSU3& l, const ScalMatSU3& r)
	{
		BinaryReturn<ScalMatSU3, ScalMatSU3, OpAdjMultiplyAdj>::Type_t  d;
	//QDPIO::cout << "ScaW=adj(ScaU)*adj(ScaV)" << endl;

		mic_MatAdjMultAdj(d.elem(), l.elem(), r.elem());

		return d;
	}


}	// namespace QDP

#endif

#endif // __MIC
