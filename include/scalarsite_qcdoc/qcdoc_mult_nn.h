#define _inline_qcdoc_mult_su3_nn(aa,bb,cc) \
{ \
  float *aptr =(float *)&(aa.elem(0,0).real().elem());  \
  float *bptr =(float *)&(bb.elem(0,0).real().elem());  \
  float *cptr =(float *)&(cc.elem(0,0).real().elem());  \
\
  __asm__ __volatile__( \
	"or    11 , %0 , %0 ;" \
	"or    12 , %1 , %1 ;" \
	"or    10 , %2 , %2 ;" \
	"li    16, 192 ;" \
	"li    17, 224 ;" \
	"li    18, 256 ;" \
\
	"lfs   0,	0( 11) ;" \
	"lfs   2,	8( 11) ;" \
	"lfs   1,	4( 11) ;" \
	"lfs   3,	12( 11) ;" \
	"lfs   6,	0( 12)  ;" \
	"lfs   7,	4( 12)  ;" \
	"lfs   8,	8( 12)  ;" \
	"lfs   9,	12( 12) ;" \
	"lfs   10,	16( 12) ;" \
	"lfs   11,	20( 12) ;" \
	\
	"fmuls   24 , 0 , 6 ;" \
	"fmuls   25 , 0 , 7 ;" \
	"lfs   12,	24( 12) ;" \
	"fmuls   26 , 0 , 8 ;" \
	"fmuls   28 , 0 , 10;" \
	"lfs   13,	28( 12) ;"\
	"fmuls   27 , 0 , 9 ;" \
	"lfs   14,	32( 12) ;" \
	"fmuls   29 , 0 , 11  ;" \
	"lfs   15,	36( 12)  ;"  \
	"fnmsubs 24 , 1 , 7 , 24 ;"   \
	"lfs   16,	40( 12) ;"   \
	"fmadds  25 , 1 , 6 , 25 ;"   \
	"lfs   17,	44( 12) ;"   \
	"fnmsubs 26 , 1 , 9 , 26 ;"   \
	"lfs   18,	48( 12) ;"   \
	"fmadds  27 , 1 , 8 , 27 ;"   \
	"lfs   19,	52( 12) ;"   \
	"fnmsubs 28 , 1 , 11 , 28 ;"  \
	"lfs   20,	56( 12) ;"   \
	"fmadds  29 , 1 , 10 , 29 ;"  \
	"lfs   21,	60( 12) ;"   \
	"fmadds  24 , 2 , 12 , 24 ;"  \
	"lfs   22,	64( 12) ;"   \
	"fmadds  25 , 2 , 13 , 25 ;"   \
	"lfs   23,	68( 12) ;"   \
	"fmadds  26 , 2 , 14 , 26 ;"   \
	"lfs   4,	16( 11) ;"   \
	"fmadds  27 , 2 , 15 , 27 ;"   \
	"lfs   5,	20( 11) ;" \
	"fmadds  28 , 2 , 16 , 28 ;" \
	"fmadds  29 , 2 , 17 , 29 ;" \
	"fnmsubs 24 , 3 , 13 , 24 ;" \
	"fmadds  25 , 3 , 12 , 25 ;" \
	"lfs   0,	24( 11) ;" \
	"fnmsubs 26 , 3 , 15 , 26 ;" \
	"fmadds  27 , 3 , 14 , 27 ;" \
	"lfs   1,	28( 11) ;" \
	"fnmsubs 28 , 3 , 17 , 28 ;" \
	"lfs   2,	32( 11) ;" \
	"fmadds  29 , 3 , 16 , 29 ;" \
	"fmadds  24 , 4 , 18 , 24 ;" \
	"fmadds  25 , 4 , 19 , 25 ;" \
	"lfs   3,	36( 11) ;" \
	"fmadds  26 , 4 , 20 , 26 ;"\
	"dcbt 16, 11 ;" \
	"fmadds  27 , 4 , 21 , 27 ;"\
	"fmadds  30 , 4 , 22 , 28 ;"\
	"fmadds  31 , 4 , 23 , 29 ;"\
	"fnmsubs 24 , 5 , 19 , 24 ;"\
	"fmadds  25 , 5 , 18 , 25 ;"\
	"lfs   4,	40( 11) ;" \
	"fnmsubs 26 , 5 , 21 , 26 ;" \
	"dcbt 	 17, 11 ;" \
	"fmadds  27 , 5 , 20 , 27 ;" \
	"fnmsubs 30 , 5 , 23 , 30 ;" \
	"fmadds  31 , 5 , 22 , 31 ;" \
	"stfs  24,	0( 10) ;  " \
	"fmuls   28 , 0 , 10 	;  " \
	"stfs  25,	4( 10) ;  " \
	"fmuls   24 , 0 , 6 	;  " \
	"fmuls   25 , 0 , 7     ;  " \
	"stfs  26,	8( 10) ;  " \
	"fmuls   29 , 0 , 11 	;  " \
	"stfs  27,	12( 10) ;  " \
	"fmuls   26 , 0 , 8 	;  " \
	"fmuls   27 , 0 , 9 	; " \
	"stfs  30,	16( 10) ;  " \
	"fnmsubs 24 , 1 , 7 , 24 ; " \
	"stfs  31,	20( 10) ;  " \
	"fmadds  25 , 1 , 6 , 25 ;  " \
	"fnmsubs 28 , 1 , 11 , 28 ;  " \
	"lfs   5,	44( 11) ;  " \
	"fnmsubs 26 , 1 , 9 , 26 ;  " \
	"dcbt 	 18, 11      	;   " \
	"fmadds  27 , 1 , 8 , 27 ;  " \
	"fmadds  29 , 1 , 10 , 29 ;  " \
	"fmadds  24 , 2 , 12 , 24 ;  " \
	"lfs   0,	48( 11) ;   " \
	"fmadds  25 , 2 , 13 , 25 ;  " \
	"dcbt 	 16, 12	;    " \
	"fmadds  26 , 2 , 14 , 26 ;  " \
	"fmadds  27 , 2 , 15 , 27 ;  " \
	"fmadds  28 , 2 , 16 , 28 ;  " \
	"lfs   1,	52( 11) ;   " \
	"fmadds  29 , 2 , 17 , 29 ;  " \
	"dcbt 	 17, 12 	;  " \
	"fnmsubs 24 , 3 , 13 , 24 ;  " \
	"fmadds  25 , 3 , 12 , 25 ;  " \
	"fnmsubs 26 , 3 , 15 , 26 ;  " \
	"lfs   2,	56( 11) ;  " \
	"fmadds  27 , 3 , 14 , 27 ;  " \
	"dcbt 	 18, 12 	;  " \
	"fnmsubs 28 , 3 , 17 , 28 ;  "\
	"fmadds  29 , 3 , 16 , 29 ;  " \
	"fmadds  24 , 4 , 18 , 24 ;  " \
	"fmadds  25 , 4 , 19 , 25 ;  " \
	"lfs   3,	60( 11) ;  " \
	"fmadds  26 , 4 , 20 , 26 ;  " \
	"fmadds  27 , 4 , 21 , 27 ;  " \
	"fmadds  30 , 4 , 22 , 28 ;  " \
	"fmadds  31 , 4 , 23 , 29 ;  " \
	"fnmsubs 24 , 5 , 19 , 24 ;  " \
	"fmadds  25 , 5 , 18 , 25 ;  " \
	"lfs   4,	64( 11) ;  " \
	"fnmsubs 26 , 5 , 21 , 26 ;  "\
	"fmadds  27 , 5 , 20 , 27 ;  "\
	"fnmsubs 30 , 5 , 23 , 30 ;  "\
	"fmadds  31 , 5 , 22 , 31 ;  "\
	"stfs  24,	24( 10) ;  "\
	"fmuls   28 , 0 , 10 	;  "\
	"stfs  25,	28( 10) ;  "\
	"fmuls   24 , 0 , 6 	;  "\
	"fmuls   25 , 0 , 7     ;  " \
	"stfs  26,	32( 10) ;  " \
	"fmuls   29 , 0 , 11 	;  " \
	"stfs  27,	36( 10) ;  "\
	"fmuls   26 , 0 , 8 	;  "\
	"fmuls   27 , 0 , 9     ;  "\
	"lfs   5,	68( 11) ;  "\
	"fnmsubs 24 , 1 , 7 , 24 ;  " \
	"stfs  30,	40( 10) ;  " \
	"fmadds  25 , 1 , 6 , 25 ;  " \
	"fnmsubs 28 , 1 , 11 , 28 ;  " \
	"stfs  31,	44( 10) ;  "\
	"fnmsubs 26 , 1 , 9 , 26 ;  "\
	"fmadds  27 , 1 , 8 , 27 ;  " \
	"fmadds  29 , 1 , 10 , 29 ; " \
	"fmadds  24 , 2 , 12 , 24 ; " \
	"fmadds  25 , 2 , 13 , 25 ; " \
	"fmadds  26 , 2 , 14 , 26 ; " \
	"fmadds  27 , 2 , 15 , 27 ; " \
	"fmadds  28 , 2 , 16 , 28 ; " \
	"fmadds  29 , 2 , 17 , 29 ; " \
	"fnmsubs 24 , 3 , 13 , 24 ; " \
	"fmadds  25 , 3 , 12 , 25 ; " \
	"fnmsubs 26 , 3 , 15 , 26 ; " \
	"fmadds  27 , 3 , 14 , 27 ; " \
	"fnmsubs 28 , 3 , 17 , 28 ; " \
	"fmadds  29 , 3 , 16 , 29 ; " \
	"fmadds  24 , 4 , 18 , 24 ; " \
	"fmadds  25 , 4 , 19 , 25 ; " \
	"fmadds  26 , 4 , 20 , 26 ; " \
	"fmadds  27 , 4 , 21 , 27 ; " \
	"fmadds  30 , 4 , 22 , 28 ; " \
	"fmadds  31 , 4 , 23 , 29 ; " \
	"fnmsubs 24 , 5 , 19 , 24 ; " \
	"fmadds  25 , 5 , 18 , 25 ; " \
	"fnmsubs 26 , 5 , 21 , 26 ; " \
	"fmadds  27 , 5 , 20 , 27 ; " \
	"fnmsubs 30 , 5 , 23 , 30 ; " \
	"fmadds  31 , 5 , 22 , 31 ; " \
	"stfs  24,	48( 10) ; " \
	"stfs  25,	52( 10) ; " \
	"stfs  26,	56( 10) ; " \
	"stfs  27,	60( 10) ; " \
	"stfs  30,	64( 10) ; " \
	"stfs  31,	68( 10) ; " \
        : : "r" (aptr), "r" (bptr), "r" (cptr)  \
	   : "16", "17", "18" ); \
}
