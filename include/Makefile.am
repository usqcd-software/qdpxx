VPATH=@top_srcdir@
VPATH+=@top_srcdir@/include
VPATH+=@top_srcdir@/include/PETE

PETE_HDRS =  PETE/Combiners.h \
             PETE/Functors.h \
             PETE/PETE.h \
             PETE/TreeNodes.h \
             PETE/CreateLeaf.h \
             PETE/ForEach.h \
             PETE/OperatorTags.h \
             PETE/Scalar.h \
             PETE/TypeComputations.h

# Generic optimization headers
genericdir = scalarsite_generic

GENERIC_HDRS = qdp_scalarsite_generic.h \
	$(genericdir)/generic_addvec.h \
	$(genericdir)/generic_adj_mat_vec.h \
	$(genericdir)/generic_mat_vec.h \
	$(genericdir)/generic_mult_aa.h \
	$(genericdir)/generic_mult_an.h \
	$(genericdir)/generic_mult_na.h \
	$(genericdir)/generic_mult_nn.h \
	$(genericdir)/qdp_scalarsite_generic_linalg.h \
	$(genericdir)/qdp_scalarsite_generic_blas.h
# If using SSE, make sure those files are installed
ssedir = scalarsite_sse

SSE_HDRS = qdp_scalarsite_sse.h \
	$(ssedir)/sse_addvec.h \
	$(ssedir)/sse_adj_mat_hwvec.h \
	$(ssedir)/sse_adj_mat_vec.h \
	$(ssedir)/sse_mat_hwvec.h \
	$(ssedir)/sse_mat_vec.h \
	$(ssedir)/sse_mult_an.h \
	$(ssedir)/sse_mult_na.h \
	$(ssedir)/sse_mult_nn.h \
	$(ssedir)/sse_s_m_a_mat.h \
	$(ssedir)/sse_s_m_a_vec.h \
	qdp_scalarvecsite_sse.h

qcdocdir = scalarsite_qcdoc

QCDOC_HDRS = qdp_scalarsite_qcdoc.h \
	$(qcdocdir)/qdp_scalarsite_qcdoc_linalg.h \
	$(qcdocdir)/qcdoc_mult_nn.h \
	$(qcdocdir)/qcdoc_mult_an.h \
	$(qcdocdir)/save_fp_regs.h

# All the include files - avoid flattening of dirs by using nobase
nobase_include_HEADERS = \
		qdp_defs.h \
		qdp_forward.h \
		qdp_globalfuncs.h \
		qdp_inner.h \
		qdp_init.h \
		qdp_io.h \
		qdp_stdio.h \
		qdp_layout.h \
		qdp_map.h \
		qdp_multi.h \
		qdp_newops.h \
		qdp_outer.h \
		qdp_outersubtype.h \
		qdp_params.h \
		qdp_primcolormat.h \
		qdp_primcolorvec.h \
		qdp_primdwvec.h \
		qdp_primgamma.h \
		qdp_primitive.h \
		qdp_primmatrix.h \
		qdp_primscalar.h \
		qdp_primseed.h \
		qdp_primspinmat.h \
		qdp_primspinvec.h \
		qdp_primvector.h \
                qdp_stopwatch.h \
		qdp_iogauge.h \
		qdp_util.h \
		qdp_xmlio.h \
		qdp_qdpexpr.h \
		qdp.h \
		qdp_qdpio.h \
		QDPOperators.h \
		qdp_qdpsubtype.h \
		qdp_qdptype.h \
		qdp_random.h \
		qdp_reality.h \
		qdp_simpleword.h \
		qdp_specializations.h \
		qdp_subset.h \
		qdp_traits.h \
		qdp_word.h \
	        qdp_scalar_specific.h \
	        qdp_parscalar_specific.h \
	        qdp_scalarvec_specific.h \
	        qdp_parscalarvec_specific.h \
		qdp_dwdefs.h \
		qdp_scalarsite_defs.h \
		qdp_scalarvecsite_defs.h \
                qdp_filebuf.h \
                $(PETE_HDRS) \
                $(GENERIC_HDRS) \
		$(QCDOC_HDRS) \
		$(SSE_HDRS)

nodist_include_HEADERS = qdp_config.h


