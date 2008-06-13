PETE_HDRS =  PETE/Combiners.h \
             PETE/Functors.h \
             PETE/PETE.h \
             PETE/TreeNodes.h \
             PETE/CreateLeaf.h \
             PETE/ForEach.h \
             PETE/ForEachInOrder.h \
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
	$(genericdir)/generic_blas_local_sumsq.h \
	$(genericdir)/generic_blas_vadd.h  \
	$(genericdir)/generic_blas_vaxmby3.h \
	$(genericdir)/generic_blas_vaxmby3_norm.h \
	$(genericdir)/generic_blas_vaxmy3.h \
	$(genericdir)/generic_blas_vaxpby3.h \
	$(genericdir)/generic_blas_vaxpy3.h \
	$(genericdir)/generic_blas_vaxpy3_norm.h \
	$(genericdir)/generic_blas_vscal.h \
	$(genericdir)/generic_blas_vsub.h \
	$(genericdir)/generic_blas_vcscal.h \
	$(genericdir)/generic_blas_vcaxpy3.h \
	$(genericdir)/generic_blas_vcaxmy3.h \
	$(genericdir)/generic_blas_vcaxpby3.h \
	$(genericdir)/generic_blas_vcaxmby3.h \
	$(genericdir)/generic_blas_local_vcdot.h \
	$(genericdir)/generic_blas_local_vcdot_real.h \
	$(genericdir)/qdp_scalarsite_generic_linalg.h \
	$(genericdir)/qdp_scalarsite_generic_blas.h \
	$(genericdir)/qdp_scalarsite_generic_cblas.h \
	$(genericdir)/qdp_scalarsite_generic_blas_g5.h \
	$(genericdir)/generic_blas_vadd3_g5.h \
	$(genericdir)/generic_blas_vscal_g5.h \
	$(genericdir)/generic_blas_vaxpy3_g5.h \
	$(genericdir)/generic_blas_vaypx3_g5.h \
	$(genericdir)/generic_blas_vaxpby3_g5.h \
	$(genericdir)/generic_blas_g5.h \
	$(genericdir)/generic_mv_switchbox.h \
	$(genericdir)/generic_spin_proj_inlines.h \
	$(genericdir)/generic_spin_recon_inlines.h \
	$(genericdir)/generic_spin_proj.h \
	$(genericdir)/generic_spin_recon.h \
	$(genericdir)/generic_fused_spin_proj.h \
	$(genericdir)/generic_fused_spin_recon.h \
	$(genericdir)/qdp_generic_spin_project_evaluates.h \
	$(genericdir)/qdp_generic_fused_spin_proj_evaluates.h \
	$(genericdir)/qdp_generic_fused_spin_recon_evaluates.h \
	$(genericdir)/generic_spin_aggregate.h


# If using SSE, make sure those files are installed
ssesitedir = scalarsite_sse

SSESITE_HDRS = qdp_scalarsite_sse.h \
	qdp_sse_intrin.h \
	$(ssesitedir)/qdp_scalarsite_sse_linalg.h \
	$(ssesitedir)/qdp_scalarsite_sse_blas.h \
	$(ssesitedir)/qdp_scalarsite_sse_blas_g5.h \
	$(ssesitedir)/qdp_scalarsite_sse_blas_g5_includes.h \
	$(ssesitedir)/sse_blas_vadd3_g5.h \
	$(ssesitedir)/sse_blas_vscal3_g5.h \
	$(ssesitedir)/sse_blas_vaxpy3_g5.h \
	$(ssesitedir)/sse_blas_vaypx3_g5.h \
	$(ssesitedir)/sse_blas_vaxpby3_g5.h \
	$(ssesitedir)/sse_spin_aggregate.h \
	$(ssesitedir)/sse_spin_proj_inlines.h \
	$(ssesitedir)/sse_spin_recon_inlines.h \
	$(ssesitedir)/sse_spin_proj.h \
	$(ssesitedir)/sse_spin_recon.h \
	$(ssesitedir)/sse_fused_spin_proj.h \
	$(ssesitedir)/sse_fused_spin_recon.h \
	$(ssesitedir)/qdp_sse_spin_evaluates.h \
	$(ssesitedir)/qdp_sse_fused_spin_proj_evaluates.h \
	$(ssesitedir)/qdp_sse_fused_spin_recon_evaluates.h \
	$(ssesitedir)/qdp_scalarsite_sse_blas_dble.h \
	$(ssesitedir)/sse_blas_vaxpy4_double.h

# Scalarvectsite SSE extensions
ssevecdir = scalarvecsite_sse

SSEVEC_HDRS = qdp_scalarvecsite_sse.h \
	$(ssevecdir)/ssevec_mult_nn.h \
	$(ssevecdir)/qdp_scalarvecsite_sse_linalg.h \
	$(ssevecdir)/qdp_scalarvecsite_sse_blas.h


# QCDOC stuff
qcdocdir = scalarsite_qcdoc

QCDOC_HDRS = $(qcdocdir)/qdp_scalarsite_qcdoc_linalg.h \
	$(qcdocdir)/qcdoc_mult_nn.h \
	$(qcdocdir)/qcdoc_mult_an.h \
	$(qcdocdir)/save_fp_regs.h


# QCDOC NONSPECIFIC BAGEL stuff
bageldir = scalarsite_bagel_qdp
BAGEL_HDRS = qdp_scalarsite_bagel_qdp.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_linalg.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_blas.h \
	$(bageldir)/qdp_scalarsite_bagel_qdp_blas_g5.h

# HEADERS for the memory allocator
MEMORY_HDRS = qdp_allocator.h \
	      qdp_singleton.h \
	      qdp_default_allocator.h \
	      qdp_qcdoc_allocator.h

# All the include files - avoid flattening of dirs by using nobase
nobase_include_HEADERS = \
		qdp_config.h \
		qdp_forward.h \
		qdp_globalfuncs.h \
		qdp_inner.h \
		qdp_init.h \
		qdp_io.h \
		qdp_binx.h \
		qdp_stdio.h \
		qdp_layout.h \
		qdp_map.h \
		qdp_multi.h \
		qdp_newops.h \
		qdp_optops.h \
		qdp_outer.h \
		qdp_outersubtype.h \
		qdp_params.h \
		qdp_precision.h \
		qdp_primcolormat.h \
		qdp_primcolorvec.h \
		qdp_primgamma.h \
		qdp_primitive.h \
		qdp_primmatrix.h \
		qdp_primscalar.h \
		qdp_primseed.h \
		qdp_primspinmat.h \
		qdp_primspinvec.h \
		qdp_primvector.h \
		qdp_profile.h \
                qdp_stopwatch.h \
		qdp_flopcount.h \
		qdp_iogauge.h \
		qdp_crc32.h \
		qdp_byteorder.h \
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
	 	qdp_defs.h \
		qdp_scalarsite_defs.h \
		qdp_scalarvecsite_defs.h \
                qdp_filebuf.h \
		qdp_strnlen.h \
                $(PETE_HDRS) \
                $(GENERIC_HDRS) \
		$(MEMORY_HDRS) \
		$(QCDOC_HDRS) \
		$(SSESITE_HDRS) \
		$(SSEVEC_HDRS) \
	        $(BAGEL_HDRS)

nodist_include_HEADERS = qdp_config_internal.h


