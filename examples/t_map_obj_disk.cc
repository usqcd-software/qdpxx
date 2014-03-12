#include "qdp.h"
#include <cstdlib>
#include <iostream>

#include "qdp_map_obj_disk.h"
#include "qdp_disk_map_slice.h"

// Including these just to check compilation
#include "qdp_map_obj_memory.h"
#include "qdp_map_obj_null.h"


namespace QDP
{
  //****************************************************************************
  //! Prop operator
  struct KeyPropColorVec_t
  {
    int        t_source;      /*!< Source time slice */
    int        colorvec_src;  /*!< Source colorvector index */
    int        spin_src;      /*!< Source spin index */
  };

  //----------------------------------------------------------------------------
  // KeyPropColorVec read
  void read(BinaryReader& bin, KeyPropColorVec_t& param)
  {
    read(bin, param.t_source);
    read(bin, param.colorvec_src);
    read(bin, param.spin_src);
  }

  // KeyPropColorVec write
  void write(BinaryWriter& bin, const KeyPropColorVec_t& param)
  {
    write(bin, param.t_source);
    write(bin, param.colorvec_src);
    write(bin, param.spin_src);
  }


  //****************************************************************************
  //! Prop operator
  struct KeyPropColorVecTimeSlice_t
  {
    int        t_source;      /*!< Source time slice */
    int        t_slice;       /*!< Time slice */
    int        colorvec_src;  /*!< Source colorvector index */
    int        spin_src;      /*!< Source spin index */
  };

  //----------------------------------------------------------------------------
  // KeyPropColorVec read
  void read(BinaryReader& bin, KeyPropColorVecTimeSlice_t& param)
  {
    read(bin, param.t_source);
    read(bin, param.t_slice);
    read(bin, param.colorvec_src);
    read(bin, param.spin_src);
  }

  // KeyPropColorVec write
  void write(BinaryWriter& bin, const KeyPropColorVecTimeSlice_t& param)
  {
    write(bin, param.t_source);
    write(bin, param.t_slice);
    write(bin, param.colorvec_src);
    write(bin, param.spin_src);
  }
}


using namespace QDP;

void fail(int line)
{
  QDPIO::cout << "FAIL: line= " << line << endl;
  QDP_finalize();
  exit(1);
}

void testMapObjInsertions(MapObjectDisk<char, float>& the_map)
{
 // Open the map for 'filling'
  //  QDPIO::cout << "Opening MapObjectDisk<char,float> for writing..."; 

  // QDPIO::cout << "Inserting (key,value): " << endl;
  // store a quadratic
  for(char i=0; i < 10; i++) { 
    float val = (float)i;
    val *= val;
    char key = 'a'+i; 

    try { 
      the_map.insert(key, val);
      QDPIO::cout << "  ( " << key <<", "<< val <<" )" << endl;
    }
    catch(...) { 
      fail(__LINE__);
    }
  }
  
  // QDPIO::cout << "Closing MapObjectDisk<char,float> for writing..." ;
}


void testMapObjLookups(MapObjectDisk<char, float>& the_map)
{
  /* Now reopen - random access */
  QDPIO::cout << "Opening MapObjectDisk<char,float> for reading..."; 

  QDPIO::cout << "Forward traversal test: " << endl;
  // Traverse in sequence
  
  QDPIO::cout << "Looking up key: " ;
  for(char i=0; i<10; i++) {
    char key = 'a'+i;
    float val;
    
    try{ 
      the_map.get(key, val);
      QDPIO::cout << " " << key;
    }
    catch(...) { 
      fail(__LINE__);
    }
    
    float refval = i*i;
    float diff = fabs(refval-val);
    if ( diff > 1.0e-8 ) {
      fail(__LINE__);
    }
    else { 
      QDPIO::cout << ".";
    }
  }

  QDPIO::cout << endl << "OK" << endl;

  QDPIO::cout << "Reverse traversal check" << endl;
  QDPIO::cout << "Looking up key: " ;
  // Traverse in reverse

  for(char i=9; i >= 0; i--) {
    char key='a'+i; 
    float val;
    try {
      the_map.get(key, val);
      QDPIO::cout << " " << key;
    }
    catch(...) { 
      fail(__LINE__);
    }

    float refval = i*i;
    float diff = fabs(refval-val);
    if ( diff > 1.0e-8 ) {
      fail(__LINE__);
    }
    else { 
      QDPIO::cout << "." << endl;
    }

  }
  QDPIO::cout << endl << "OK" << endl;


  // 20 'random' reads
  QDPIO::cout << "Random access... 20 reads" << endl;
  QDPIO::cout << "Looking up key: " ;
  for(int j=0; j < 20; j++){ 
    char key = 'a'+std::rand() % 10; // Pick randomly in 0..10
    float val;

    try { 
      the_map.get(key, val);
      QDPIO::cout << " " << key ;
    }
    catch(...) {
      fail(__LINE__);
    }

    float refval = (key-'a')*(key-'a');

    float diff = fabs(refval-val);
    if ( diff > 1.0e-8 ) {
      fail(__LINE__);
    }
    else { 
      QDPIO::cout <<".";
    }
  }
  QDPIO::cout << endl << "OK" << endl;
}



//**********************************************************************************************
void testMapKeyPropColorVecInsertions(MapObjectDisk<KeyPropColorVec_t, LatticeFermion>& pc_map, 
				      const multi1d<LatticeFermion>& lf_array)
{
  // Create the key-type
  KeyPropColorVec_t the_key = {0,0,0};

  // OpenMap for Writing
  QDPIO::cout << "Opening Map<KeyPropColorVec_t,LF> for writing..." << endl;
  QDPIO::cout << "Currently map has size = " << pc_map.size() << endl;

  QDPIO::cout << "Inserting array element : ";
  StopWatch swatch;
  swatch.reset();
  swatch.start();

  for(int i=0; i < lf_array.size(); i++) { 
    the_key.colorvec_src = i;

    try { 
      if (pc_map.insert(the_key, lf_array[i]) != 0)
      {
	QDPIO::cerr << __func__ << ": error writing key\n";
	QDP_abort(1);
      }
      QDPIO::cout << " "<< i << endl;
    }
    catch(...) {
      fail(__LINE__);
    }
  }
  swatch.stop();
  double time = swatch.getTimeInSeconds();

  QDPIO::cout << "Number of seconds to write = " << time << endl;

  QDPIO::cout << "Currently map has size = " << pc_map.size() << endl;
  QDPIO::cout << "Closing Map<KeyPropColorVec_t,LF> for writing..." << endl;
}


void testMapKeyPropColorVecLookups(MapObjectDisk<KeyPropColorVec_t, LatticeFermion>& pc_map, 
				   const multi1d<LatticeFermion>& lf_array)
{
  // Open map in read mode
  QDPIO::cout << "Opening Map<KeyPropColorVec_t,LF> for reading.." << endl;
  QDPIO::cout << "Currently map has size = " << pc_map.size() << endl;

  QDPIO::cout << "Increasing lookup test:" << endl;
  QDPIO::cout << "Looking up with colorvec_src = ";
  // Create the key-type
  KeyPropColorVec_t the_key = {0,0,0};

  StopWatch swatch;
  swatch.reset();
  swatch.start();

  for(int i=0; i < lf_array.size(); i++) {
    LatticeFermion lf_tmp;

    the_key.colorvec_src=i;
    try{
      if (pc_map.get(the_key, lf_tmp) != 0)
      {
	QDPIO::cerr << __func__ << ": error in get\n";
	QDP_abort(1);
      }
      QDPIO::cout << " " << i;
    }
    catch(...) { 
      fail(__LINE__);
    }

#if 1
    // Compare with lf_array
    LatticeFermion diff;
    diff = lf_tmp - lf_array[i];
    Double diff_norm = sqrt(norm2(diff))/Double(Nc*Ns*Layout::vol());
    if(  toDouble(diff_norm) < 1.0e-6 )  { 
      QDPIO::cout << "." ;
    }
    else { 
      fprintf (stderr, "Difference between retrieved value and expected is %lf\n",
	       toDouble(diff_norm));
      fail(__LINE__);
    }
#endif

  }
  QDPIO::cout << endl << "OK" << endl;

  swatch.stop();
  double time = swatch.getTimeInSeconds();

  QDPIO::cout << "Number of seconds to read = " << time << endl;

#if 1
  QDPIO::cout << "Random access lookup test" << endl;
  QDPIO::cout << "Looking up with colorvec_src = " ;

  swatch.reset();
  swatch.start();

  // Hey DJ! Spin that disk...
  for(int j=0; j < 100; j++) {
    int i = random() % lf_array.size();
    LatticeFermion lf_tmp;
    the_key.colorvec_src=i;
    try{
      if (pc_map.get(the_key, lf_tmp) != 0)
      {
	QDPIO::cerr << __func__ << ": error in get\n";
	QDP_abort(1);
      }
      QDPIO::cout << " " << i;
    }
    catch(...) {
      fail(__LINE__);
    }

    // Compare with lf_array
    LatticeFermion diff;
    diff = lf_tmp - lf_array[i];
    Double diff_norm = sqrt(norm2(diff))/Double(Nc*Ns*Layout::vol());
    if(  toDouble(diff_norm) < 1.0e-6 )  { 
      QDPIO::cout << ".";
    }
    else { 
      fail(__LINE__);
    }
  }

  swatch.stop();
  time = swatch.getTimeInSeconds();
  QDPIO::cout << "Number of seconds to random read = " << time << endl;
#endif

  QDPIO::cout << "Currently map has size = " << pc_map.size() << endl;
  QDPIO::cout << endl << "OK" << endl;
}


//**********************************************************************************************
void testMapKeyPropColorVecInsertionsTimeSlice(MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> >& pc_map, 
					       const multi1d<LatticeFermion>& lf_array)

{
  // Create the key-type
  KeyPropColorVecTimeSlice_t the_key = {0,0,0,0};

  // OpenMap for Writing
  QDPIO::cout << "Opening Map<KeyPropColorVecTimeSlice_t,TimeSlice<LF>> for writing..." << endl;
  QDPIO::cout << "Currently map has size = " << pc_map.size() << endl;

  QDPIO::cout << "Inserting array element : ";

  StopWatch swatch;
  swatch.reset();
  swatch.start();

  for(int i=0; i < lf_array.size(); i++) { 
    the_key.colorvec_src = i;

    try { 
      for(int time_slice=0; time_slice < Layout::lattSize()[Nd-1]; ++time_slice)
      {
	the_key.t_slice = time_slice;

	LatticeFermion fred = lf_array[i];
	if (pc_map.insert(the_key, TimeSliceIO<LatticeFermion>(fred, time_slice)) != 0)
	{
	  QDPIO::cerr << __func__ << ": error writing key\n";
	  QDP_abort(1);
	}
	// QDPIO::cout << "i= "<< i << "  time_slice= " << time_slice << endl;
      }
    }
    catch(...) {
      fail(__LINE__);
    }
  }
  swatch.stop ();
  double time = swatch.getTimeInSeconds();

  QDPIO::cout << "Number of seconds to time slice write = " << time << endl;

  QDPIO::cout << "Before exiting map has size = " << pc_map.size() << endl;
  QDPIO::cout << "Finishing Map<KeyPropColorVecTimeSlice_t,TimeSlice<LF>> for writing..." << endl;
}


void testMapKeyPropColorVecLookupsTimeSlice(MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> >& pc_map, 
					    const multi1d<LatticeFermion>& lf_array)
{
  // Open map in read mode
  QDPIO::cout << "Opening Map<KeyPropColorVecTimeSlice_t,TimeSlice<LF>> for reading.." << endl;
  QDPIO::cout << "Before starting map has size = " << pc_map.size() << endl;

  QDPIO::cout << "Increasing lookup test:" << endl;
  QDPIO::cout << "Looking up with colorvec_src = ";
  // Create the key-type
  KeyPropColorVecTimeSlice_t the_key = {0,0,0,0};

  StopWatch swatch;
  swatch.reset();
  swatch.start();

  for(int i=0; i < lf_array.size(); i++) {
    LatticeFermion lf_tmp;

    the_key.colorvec_src=i;
    try{
      for(int time_slice=0; time_slice < Layout::lattSize()[Nd-1]; ++time_slice)
      {
	the_key.t_slice = time_slice;

	TimeSliceIO<LatticeFermion> time_slice_lf(lf_tmp, time_slice);
	if (pc_map.get(the_key, time_slice_lf) != 0)
	{
	  QDPIO::cerr << __func__ << ": error in get\n";
	  QDP_abort(1);
	}
	// QDPIO::cout << "i= "<< i << "  time_slice= " << time_slice << endl;
      }
    }
    catch(...) { 
      fail(__LINE__);
    }

    // Compare with lf_array
    LatticeFermion diff;
    diff = lf_tmp - lf_array[i];
    Double diff_norm = sqrt(norm2(diff))/Double(Nc*Ns*Layout::vol());
    if(  toDouble(diff_norm) < 1.0e-6 )  { 
      QDPIO::cout << "." ;
    }
    else { 
      QDPIO::cout << "norm2(diff)= " << diff_norm << endl;
      fail(__LINE__);
    }

  }
  QDPIO::cout << endl << "OK" << endl;

  swatch.stop ();
  double time = swatch.getTimeInSeconds();

  QDPIO::cout << "Number of seconds to time slice read = " << time << endl;

#if 1
  QDPIO::cout << "Random access lookup test" << endl;
  QDPIO::cout << "Looking up with colorvec_src = " ;
  // Hey DJ! Spin that disk...
  for(int j=0; j < 20; j++) {
    int i = random() % lf_array.size();
    LatticeFermion lf_tmp = zero;
    the_key.colorvec_src=i;
    try{
      for(int time_slice=0; time_slice < Layout::lattSize()[Nd-1]; ++time_slice)
      {	
	the_key.t_slice = time_slice;

	TimeSliceIO<LatticeFermion> time_slice_lf(lf_tmp, time_slice);
	if (pc_map.get(the_key, time_slice_lf) != 0)
	{
	  QDPIO::cerr << __func__ << ": error in get\n";
	  QDP_abort(1);
	}
	QDPIO::cout << "i= "<< i << "  time_slice= " << time_slice << endl;
      }
    }
    catch(...) {
      fail(__LINE__);
    }

    // Compare with lf_array
    LatticeFermion diff;
    diff = lf_tmp - lf_array[i];
    Double diff_norm = sqrt(norm2(diff))/Double(Nc*Ns*Layout::vol());
    if(  toDouble(diff_norm) < 1.0e-6 )  { 
      QDPIO::cout << ".";
    }
    else { 
      QDPIO::cout << "norm2(diff)= " << diff_norm << endl;
      fail(__LINE__);
    }
  }
#endif
  QDPIO::cout << endl << "OK" << endl;
}



//**********************************************************************************************
int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  multi1d<int> nrow(Nd);

  if (argc >= 5) {
    nrow[0] = atoi(argv[1]);
    nrow[1] = atoi(argv[2]);
    nrow[2] = atoi(argv[3]);
    nrow[3] = atoi(argv[4]);
    
    QDP_info ("Lattice size %dx%dx%dx%d\n", nrow[0], nrow[1], nrow[2], nrow[3]);
  }
  else {
    // Setup the layout
    // const int foo[] = {4,2,2,2};
    // const int foo[] = {16,2,16,2};
    // const int foo[] = {32, 32, 32, 32};
    const int foo[] = {4, 4, 4, 2};
    nrow = foo;  // Use only Nd elements
  }

  Layout::setLattSize(nrow);
  Layout::create();

  // For debugging
  {
    MapObjectNull<char,float> dumb;
  }

  // For debugging
  {
    MapObjectMemory<char,float> dumb;
  }

  string map_obj_file;
  if (argc >= 6) {
    map_obj_file = argv[5];
  }
  else {
    // Params to create a map object disk
    // string map_obj_f("/scratch/chen/t_map_obj_disk.mod");
    string map_obj_f("/volatile/users/chen/piotest/t_map_obj_disk.mod");
    // string map_obj_f ("/dev/null");
    map_obj_file = map_obj_f;
  }

  // Some metadata
  string meta_data;
  {
    XMLBufferWriter file_xml;

    push(file_xml, "DBMetaData");
    write(file_xml, "id", string("propElemOp"));
    write(file_xml, "lattSize", QDP::Layout::lattSize());
    pop(file_xml);

    meta_data = file_xml.str();
  }

#if 1
  //
  // Test simple scalar
  //
  try {
    // Make a disk map object -- keys are ints, data floats
    MapObjectDisk<char,float> made_map;
    // made_map.setDebug(10);
    made_map.insertUserdata(meta_data);
    made_map.open(map_obj_file, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);

    testMapObjInsertions(made_map);
    testMapObjLookups(made_map);
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

  // Make an array of LF-s filled with noise
  multi1d<LatticeFermion> lf_array(10);
    
  for(int i=0; i < lf_array.size(); i++) { 
    gaussian(lf_array[i]);
  }


#if 1
  //
  // Test lattice objects
  //
  try {
    QDPIO::cout << "\n\n\nTest DB over a lattice" << endl;

    MapObjectDisk<KeyPropColorVec_t, LatticeFermion> pc_map;

    // pc_map.setDebug(2);

    pc_map.insertUserdata(meta_data);
    pc_map.open(map_obj_file, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
    testMapKeyPropColorVecInsertions(pc_map, lf_array);
    testMapKeyPropColorVecLookups(pc_map, lf_array);
    QDPIO::cout << endl << "OK OK ....." << endl;

#if 1
    // Test an update 
    QDPIO::cout << "Doing update test ..." << endl;
    KeyPropColorVec_t the_key = {0,0,0};
    the_key.colorvec_src = 5;
    LatticeFermion f; gaussian(f);
    QDPIO::cout << "Updating..." ;
    if (pc_map.insert(the_key,f) != 0)
    {
      QDPIO::cerr << __func__ << ": error writing key\n";
      QDP_abort(1);
    }
    QDPIO::cout << "OK" << endl;


    LatticeFermion f2;
    QDPIO::cout << "Re-Looking up...";
    if (pc_map.get(the_key,f2) != 0)
    {
      QDPIO::cerr << __func__ << ": error in get\n";
      QDP_abort(1);
    }
    QDPIO::cout << "OK" << endl;

    QDPIO::cout << "Comparing..." << endl;
    f2 -= f;
    if( toBool( sqrt(norm2(f2)) > toDouble(1.0e-6) ) ) {
      QDPIO::cout << "sqrt(norm2(f2))=" << sqrt(norm2(f2)) << endl;
      fail(__LINE__);
    }
    else { 
      QDPIO::cout << "OK" << endl ;
    }
    // Reinsert previous value
    if (pc_map.insert(the_key,lf_array[5]) != 0)
    {
      QDPIO::cerr << __func__ << ": error inserting\n";
      QDP_abort(1);
    }
    testMapKeyPropColorVecLookups(pc_map, lf_array);
#endif
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

#if 1
  //
  // Test with time slices
  //
  try {
    QDPIO::cout << "\n\n\nTest DB with time-slices" << endl;

    MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> > pc_map;
    //    pc_map.setDebug(1);
    pc_map.setDebug(0);
    pc_map.insertUserdata(meta_data);
    pc_map.open(map_obj_file, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
    
    int Lt = Layout::lattSize()[Nd-1];
    testMapKeyPropColorVecInsertionsTimeSlice(pc_map, lf_array);
    testMapKeyPropColorVecLookupsTimeSlice(pc_map, lf_array);
    QDPIO::cout << endl << "OK" << endl;

    // Test an update 
    QDPIO::cout << "Doing update test ..." << endl;
    KeyPropColorVecTimeSlice_t the_key = {0,0,0,0};
    the_key.colorvec_src = 5;
    LatticeFermion f; gaussian(f);
    QDPIO::cout << "Updating..." ;
    for(int time_slice=0; time_slice < Lt; ++time_slice)
    {
      the_key.t_slice = time_slice;
      TimeSliceIO<LatticeFermion> time_slice_f(f, time_slice);
      if (pc_map.insert(the_key,time_slice_f) != 0)
      {
	QDPIO::cerr << __func__ << ": error inserting\n";
	QDP_abort(1);
      }
    }
    QDPIO::cout << "OK" << endl;

    LatticeFermion f2;
    QDPIO::cout << "Re-Looking up...";
    for(int time_slice=0; time_slice < Lt; ++time_slice)
    {
      the_key.t_slice = time_slice;
      TimeSliceIO<LatticeFermion> time_slice_f2(f2, time_slice);
      if (pc_map.get(the_key,time_slice_f2) != 0)
      {
	QDPIO::cerr << __func__ << ": error in get\n";
	QDP_abort(1);
      }
    }
    QDPIO::cout << "OK" << endl;

    QDPIO::cout << "Comparing..." << endl;
    f2 -= f;
    if( toBool( sqrt(norm2(f2)) > toDouble(1.0e-6) ) ) {
      QDPIO::cout << "sqrt(norm2(f2))=" << sqrt(norm2(f2)) << endl;
      fail(__LINE__);
    }
    else { 
      QDPIO::cout << "OK" << endl ;
    }
    // Reinsert previous value
    LatticeFermion lf5 = lf_array[5];
    for(int time_slice=0; time_slice < Lt; ++time_slice)
    {
      the_key.t_slice = time_slice;
      TimeSliceIO<LatticeFermion> time_slice_lf5(lf5, time_slice);
      if (pc_map.insert(the_key,time_slice_lf5) != 0)
      {
	QDPIO::cerr << __func__ << ": error inserting\n";
	QDP_abort(1);
      }
    }
    testMapKeyPropColorVecLookupsTimeSlice(pc_map, lf_array);
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

#if 1
  //
  // Test inserting more into previous DB
  //
  try {
    QDPIO::cout << "\n\n\nTest inserting more time-slice data into previous DB" << endl;

    MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> > pc_map;
    pc_map.setDebug(1);
    pc_map.open(map_obj_file, std::ios_base::in | std::ios_base::out);
    
    int Lt = Layout::lattSize()[Nd-1];

    KeyPropColorVecTimeSlice_t the_key = {0,0,0,0};
    the_key.colorvec_src = 17;

    LatticeFermion f; gaussian(f);

    for(int time_slice=0; time_slice < Lt; ++time_slice)
    {
      the_key.t_slice = time_slice;
      if (pc_map.insert(the_key, TimeSliceIO<LatticeFermion>(f, time_slice)) != 0)
      {
	QDPIO::cerr << __func__ << ": error inserting\n";
	QDP_abort(1);
      }
    }
    QDPIO::cout << endl << "OK" << endl;

    QDPIO::cout << endl << "Check flushing the db..." << endl;
    pc_map.flush();

    QDPIO::cout << endl << "OK" << endl;
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

#if 1
  //
  // Test read mode on previous DB
  //
  try {
    QDPIO::cout << "\n\n\nTest reading previous DB with time-slices" << endl;

    MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> > pc_map;

    pc_map.setDebug(1);
    pc_map.open(map_obj_file, std::ios_base::in);
    
    int Lt = Layout::lattSize()[Nd-1];
    testMapKeyPropColorVecLookupsTimeSlice(pc_map, lf_array);
    QDPIO::cout << endl << "OK" << endl;
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

#if 1
  //
  // Test stuff
  //
  try {
    QDPIO::cout << "\n\n\nTest file names" << endl;

    QDPIO::cout << "Check for map_obj_file(in) : status= " << MapObjDiskEnv::checkForNewFile(map_obj_file, std::ios_base::in) << "\n";
    QDPIO::cout << "Check for map_obj_file(trunc) : status= " << MapObjDiskEnv::checkForNewFile(map_obj_file, std::ios_base::in | std::ios_base::trunc) << "\n";

    QDPIO::cout << "Check for fred.foo(in) : status= " << MapObjDiskEnv::checkForNewFile("fred.foo", std::ios_base::in) << "\n";
    QDPIO::cout << "Check for fred.foo(trunc) : status= " << MapObjDiskEnv::checkForNewFile("fred.foo", std::ios_base::in | std::ios_base::trunc) << "\n";

    MapObjectDisk<KeyPropColorVecTimeSlice_t, TimeSliceIO<LatticeFermion> > pc_map;

    QDPIO::cout << "Check for map_obj_file(in) : exist= " << pc_map.fileExists(map_obj_file) << "\n";
    QDPIO::cout << "Check for fred.foo(in) : exist= " << pc_map.fileExists("fred.foo") << "\n";
  }
  catch(const std::string& e) { 
    QDPIO::cout << "Caught: " << e << endl;
    fail(__LINE__);
  }
#endif

  QDP_finalize();
  return 0;
}

