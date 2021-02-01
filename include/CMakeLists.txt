set( PETE_HDRS PETE/Combiners.h
             PETE/Functors.h
             PETE/PETE.h
             PETE/TreeNodes.h
             PETE/CreateLeaf.h
             PETE/ForEach.h
             PETE/ForEachInOrder.h
             PETE/OperatorTags.h
             PETE/Scalar.h
             PETE/TypeComputations.h)
             
install(FILES ${PETE_HDRS} DESTINATION include/PETE)


set(GENERIC_HDRS scalarsite_generic/generic_addvec.h
	scalarsite_generic/generic_adj_mat_vec.h
	scalarsite_generic/generic_mat_vec.h
	scalarsite_generic/generic_mult_aa.h
	scalarsite_generic/generic_mult_an.h
	scalarsite_generic/generic_mult_na.h
	scalarsite_generic/generic_mult_nn.h
	scalarsite_generic/generic_blas_local_sumsq.h
	scalarsite_generic/generic_blas_vadd.h
	scalarsite_generic/generic_blas_vaxmby3.h
	scalarsite_generic/generic_blas_vaxmby3_norm.h
	scalarsite_generic/generic_blas_vaxmy3.h
	scalarsite_generic/generic_blas_vaxpby3.h
	scalarsite_generic/generic_blas_vaxpy3.h
	scalarsite_generic/generic_blas_vaxpy3_norm.h
	scalarsite_generic/generic_blas_vscal.h
	scalarsite_generic/generic_blas_vsub.h
	scalarsite_generic/generic_blas_vcscal.h
	scalarsite_generic/generic_blas_vcaxpy3.h
	scalarsite_generic/generic_blas_vcaxmy3.h
	scalarsite_generic/generic_blas_vcaxpby3.h
	scalarsite_generic/generic_blas_vcaxmby3.h
	scalarsite_generic/generic_blas_local_vcdot.h
	scalarsite_generic/generic_blas_local_vcdot_real.h
	scalarsite_generic/qdp_scalarsite_generic_linalg.h
	scalarsite_generic/qdp_scalarsite_generic_blas.h
	scalarsite_generic/qdp_scalarsite_generic_cblas.h
	scalarsite_generic/qdp_scalarsite_generic_blas_g5.h
	scalarsite_generic/generic_blas_vadd3_g5.h
	scalarsite_generic/generic_blas_vscal_g5.h
	scalarsite_generic/generic_blas_vaxpy3_g5.h
	scalarsite_generic/generic_blas_vaypx3_g5.h
	scalarsite_generic/generic_blas_vaxpby3_g5.h
	scalarsite_generic/generic_blas_g5.h
	scalarsite_generic/generic_mv_switchbox.h
	scalarsite_generic/generic_spin_proj_inlines.h
	scalarsite_generic/generic_spin_recon_inlines.h
	scalarsite_generic/generic_spin_proj.h
	scalarsite_generic/generic_spin_recon.h
	scalarsite_generic/generic_fused_spin_proj.h
	scalarsite_generic/generic_fused_spin_recon.h
	scalarsite_generic/qdp_generic_spin_project_evaluates.h
	scalarsite_generic/qdp_generic_fused_spin_proj_evaluates.h
	scalarsite_generic/qdp_generic_fused_spin_recon_evaluates.h
	scalarsite_generic/generic_spin_aggregate.h
	scalarsite_generic/qdp_scalarsite_generic_blas_wrapper.h
	scalarsite_generic/qdp_scalarsite_generic_cblas_wrapper.h
	scalarsite_generic/qdp_scalarsite_generic_blas_g5_wrapper.h
	scalarsite_generic/qdp_generic_spin_project_evaluates_wrapper.h
	scalarsite_generic/qdp_generic_fused_spin_proj_evaluates_wrapper.h
	scalarsite_generic/qdp_generic_fused_spin_recon_evaluates_wrapper.h
	scalarsite_generic/qdp_scalarsite_generic_linalg_wrapper.h
)

install(FILES qdp_scalarsite_generic.h DESTINATION include)
install(FILES ${GENERIC_HDRS} DESTINATION include/scalarsite_generic)


set(SSESITE_HDRS scalarsite_sse/qdp_scalarsite_sse_linalg.h
	scalarsite_sse/qdp_scalarsite_sse_blas.h
	scalarsite_sse/qdp_scalarsite_sse_blas_g5.h
	scalarsite_sse/qdp_scalarsite_sse_blas_g5_includes.h
	scalarsite_sse/sse_blas_vadd3_g5.h
	scalarsite_sse/sse_blas_vscal3_g5.h
	scalarsite_sse/sse_blas_vaxpy3_g5.h
	scalarsite_sse/sse_blas_vaypx3_g5.h
	scalarsite_sse/sse_blas_vaxpby3_g5.h
	scalarsite_sse/sse_spin_aggregate.h
	scalarsite_sse/sse_spin_proj_inlines.h
	scalarsite_sse/sse_spin_recon_inlines.h
	scalarsite_sse/sse_spin_proj.h
	scalarsite_sse/sse_spin_recon.h
	scalarsite_sse/sse_fused_spin_proj.h
	scalarsite_sse/sse_fused_spin_recon.h
	scalarsite_sse/qdp_sse_spin_evaluates.h
	scalarsite_sse/qdp_sse_fused_spin_proj_evaluates.h
	scalarsite_sse/qdp_sse_fused_spin_recon_evaluates.h
	scalarsite_sse/qdp_scalarsite_sse_blas_double.h
	scalarsite_sse/sse_blas_local_sumsq_double.h
	scalarsite_sse/sse_blas_local_vcdot_double.h
	scalarsite_sse/sse_blas_local_vcdot_real_double.h
	scalarsite_sse/sse_blas_vaxmbyz4_double.h
	scalarsite_sse/sse_blas_vaxpbyz4_double.h
	scalarsite_sse/sse_blas_vaxmyz4_double.h
	scalarsite_sse/sse_blas_vaxpy4_double.h
	scalarsite_sse/sse_blas_vaypx4_double.h
	scalarsite_sse/sse_blas_vscal4_double.h
	scalarsite_sse/sse_linalg_mm_su3_double.h
	scalarsite_sse/qdp_scalarsite_sse_linalg_double.h
	scalarsite_sse/sse_dcomplex_mult_macros.h
	scalarsite_sse/sse_prefetch.h
	scalarsite_sse/qdp_scalarsite_sse_blas_wrapper.h
	scalarsite_sse/qdp_scalarsite_sse_blas_double_wrapper.h
	scalarsite_sse/qdp_sse_spin_evaluates_wrapper.h
	scalarsite_sse/qdp_sse_fused_spin_proj_evaluates_wrapper.h
	scalarsite_sse/qdp_sse_fused_spin_recon_evaluates_wrapper.h
)

install(FILES qdp_scalarsite_sse.h
	          qdp_sse_intrin.h DESTINATION include)
install(FILES ${SSESITE_HDRS} DESTINATION include/scalarsite_sse)


set(SSEVEC_HDRS scalarvecsite_sse/ssevec_mult_nn.h
	scalarvecsite_sse/qdp_scalarvecsite_sse_linalg.h
	scalarvecsite_sse/qdp_scalarvecsite_sse_blas.h
)

install(FILES qdp_scalarvecsite_sse.h DESTINATION include)
install(FILES ${SSEVEC_HDRS} DESTINATION include/scalarvecsite_sse)

set( MEMORY_HDRS qdp_allocator.h 
	qdp_singleton.h
	qdp_default_allocator.h
	qdp_pool_allocator.h
)
install(FILES ${MEMORY_HDRS} DESTINATION include)
	
set( QDP_HDRS qdp_config.h
		qdp_diagnostics.h
		qdp_forward.h
		qdp_globalfuncs.h
		qdp_globalfuncs_subtype.h
		qdp_inner.h
		qdp_init.h
		qdp_io.h
		qdp_stdio.h
		qdp_layout.h
		qdp_map.h
		qdp_multi.h
		qdp_arrays.h
		qdp_newops.h
		qdp_optops.h
		qdp_outer.h
		qdp_outersubtype.h
		qdp_params.h
		qdp_precision.h
		qdp_primcolormat.h
		qdp_primcolorvec.h
		qdp_primgamma.h
		qdp_primitive.h
		qdp_primmatrix.h
		qdp_primscalar.h
		qdp_primseed.h
		qdp_primspinmat.h
		qdp_primspinvec.h
		qdp_primvector.h
		qdp_profile.h
      	qdp_stopwatch.h
		qdp_flopcount.h
		qdp_iogauge.h
		qdp_crc32.h
		qdp_byteorder.h
		qdp_util.h
		qdp_xmlio.h
		qdp_qdpexpr.h
		qdp.h
		qdp_qdpio.h
		QDPOperators.h
		qdp_qdpsubtype.h
		qdp_qdptype.h
		qdp_random.h
		qdp_rannyu.h
		qdp_reality.h
		qdp_simpleword.h
		qdp_specializations.h
		qdp_subset.h
		qdp_traits.h
		qdp_word.h
		qdp_dispatch.h
	    qdp_scalar_specific.h
	    qdp_parscalar_specific.h
	    qdp_scalarvec_specific.h
	    qdp_parscalarvec_specific.h
	 	qdp_defs.h
		qdp_scalarsite_defs.h
		qdp_scalarvecsite_defs.h
		qdp_strnlen.h
		qdp_db.h
		qdp_db_imp.h
		qdp_db_stub.h
		qdp_map_obj.h
		qdp_map_obj_null.h
		qdp_map_obj_memory.h
		qdp_map_obj_disk.h
		qdp_map_obj_disk_multiple.h
		qdp_disk_map_slice.h
		qdp_hdf5.h
		qdp_threadbind.h
)

install( FILES ${QDP_HDRS} DESTINATION include ) 