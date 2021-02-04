add_library(examples mesplq.cc reunit.cc dslashm_w.cc rgauge.cc taproj.cc 
			expm12.cc baryon_w.cc mesons_w.cc formfac_w.cc linalg1.cc linalg.h 
			blas1.cc blas1.h cblas1.cc cblas1.h)
			
target_link_libraries(examples PUBLIC qdp)



set(TESTS t_skeleton 
	t_foo 
	t_xml 
	t_qio 
	t_qio_factory 
	t_partfile
	t_basic 
	t_shift 
	t_io 
	t_entry 
	t_nersc 
	t_dslashm 
	t_mesplq
    t_exotic 
    t_formfac 
    t_spectrum 
    t_qdp 
    t_linalg 
    t_blas 
    t_iprod
    t_cblas 
    t_clov_force 
    t_db 
    t_map_obj_disk 
    t_map_obj_memory 
    t_blas_g5 
    t_blas_g5_2 
    t_blas_g5_3 
    t_spinproj 
    t_spinproj2
    t_cugauge 
    t_transpose_spin 
    nersc2ildg 
    test_ildglat 
    t_su3
    lhpc2ildg 
)

if( QDP_USE_SSE )
	set(TESTS ${TESTS} t_gsum)
endif()

foreach(TEST ${TESTS})
 add_executable(${TEST} ${TEST}.cc)
 target_link_libraries(${TEST} examples)
endforeach()

