
set(test_HDRS unittest.h testvol.h)

add_executable(test_vaxpy_double
	${test_HDRS}
	testVaypxDouble.h
	testVaypxDouble.cc
	testVaxpyDouble.h
	testVaxpyDouble.cc
	testVaxmyDouble.h
	testVaxmyDouble.cc
	testVaxpbyDouble.h
	testVaxpbyDouble.cc
	testVScalDouble.h
	testVScalDouble.cc
	testLocalSumSqDouble.h
	testLocalSumSqDouble.cc
	testLocalVcdotRealDouble.h
	testLocalVcdotRealDouble.cc
	testLocalVcdotDouble.h
	testLocalVcdotDouble.cc
	test_vaxpy_double.cc)
target_link_libraries(test_vaxpy_double PUBLIC qdp)


add_executable(time_vaxpy_double
	${test_HDRS}
	timeVaxpyDouble.h
	timeVaxpyDouble.cc
	timeVaxpby.cc
	time_vaxpy_double.cc)
target_link_libraries(time_vaxpy_double PUBLIC qdp)	

add_executable(test_cmul
	test_cmul.cc
	testCMulDouble.h
	testCMulDouble.cc)
target_link_libraries(test_cmul PUBLIC qdp)

add_executable(test_matmat_double
	test_matmat_double.cc
	testMatScalMultDouble.h
	testMatScalMultDouble.cc
	testMatPeqMatDouble.h
	testMatPeqMatDouble.cc
	testMatEqMatMatDouble.h
	testMatEqMatMatDouble.cc
	testMatEqMatHermDouble.h
	testMatEqMatHermDouble.cc
	testMatEqHermMatDouble.h
	testMatEqHermMatDouble.cc
	testMatEqHermHermDouble.h
	testMatEqHermHermDouble.cc)
target_link_libraries(test_matmat_double PUBLIC qdp)	


add_executable(time_matmat_double 
	time_matmat_double.cc
	timeMatEqMatMatDouble.h
	timeMatEqMatMatDouble.cc
	timeMatEqMatHermDouble.h
	timeMatEqMatHermDouble.cc
	timeMatEqHermMatDouble.h
	timeMatEqHermMatDouble.cc
	timeMatEqHermHermDouble.h
	timeMatEqHermHermDouble.cc)
target_link_libraries(time_matmat_double PUBLIC qdp)
	

