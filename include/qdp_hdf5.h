// -*- C++ -*-

/*! @file
* @brief HDF5 support
*
* Functions for reading/writing files with hdf5

*/

#ifndef QDP_HDF5_H
#define QDP_HDF5_H

#include <string>
#include <sstream>
#include <stack>
#include <list>

#include <hdf5.h>

//qdp related things:
#include "qdp_util.h"
#include "qdp_stopwatch.h"

typedef unsigned long long ullong;

namespace QDP {

    
	//--------------------------------------------------------------------------------                                                                 
	//! HDF5 reader class                                                                                                                              
	/*!                                                                                                                                                
	This is used to read data from an HDF5 file using.                                                                                               
	*/
	class HDF5
	{
	protected:
		hid_t error_stack, file_id, file_comm, current_group;
		
		//Lustre optimizations
		long int stripesize, maxalign;
			
		//oother stuff:
		bool par_init;
		bool profile;
		//copy of qmp mpi-communicator
		MPI_Comm* mpicomm;

		//constructors:
		HDF5();
		HDF5(const long int&, const long int& maxalign=0);

		//stack with all open groups:                                                                                                                    
		std::string getNameById(hid_t id)const;

		//helper for splitting directory names
		std::vector<std::string> splitPathname(const std::string& name);

		//check if an object exists, by iterating through the tree:
		bool objectExists(hid_t loc_id, const std::string& name);

		//error handler
		static hid_t errorHandler(hid_t errstack, void* unused);

		void HDF5_error_exit(const std::string& message){
			close();
			QDP_error_exit(message.c_str());
		}

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//DATATYPE HELPERS
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//create complex
		hid_t createComplexType(const unsigned int& float_size);

		//check complex:
		bool checkComplexType(const hid_t& type_id, hid_t& base_type_id);

		//create colormat
		hid_t createColorMatrixType(const hid_t& complex_id, const unsigned int& rank);

		//check colormat:
		bool checkColorMatrixType(const hid_t& type_id, const unsigned int& rank, hid_t& base_type_id);

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING ATTRIBUTES HELPERS                                                                                                         
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//private read helper routines
		template<typename ctype>
		void rdAtt(const std::string& obj_name, const std::string& attr_name, ctype& datum, const H5T_class_t& hdfclass, const bool& sign, const bool& rigid_checks=true){
			//get datatype properties:
			unsigned int size=sizeof(ctype);

			std::string oname(obj_name), aname(attr_name);
			bool exists=objectExists(current_group,oname);
			if(!exists){
				HDF5_error_exit("HDF5Reader::readAttribute: error, object "+oname+" you try to read attribute from does not exists!");
			}

			//do sanity checks and get datatype 
			hid_t ex=H5Aexists_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			if(ex!=1){
				HDF5_error_exit("HDF5Reader::readAttribute: error, the attribute "+aname+" you try to read does not exists!");
			}
			hid_t attr_id=H5Aopen_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT,H5P_DEFAULT);
			if(attr_id<0){
				HDF5_error_exit("HDF5Reader::readAttribute: error, cannot open attribute "+aname+" attached to "+oname+"!");
			}
			hid_t type_id=H5Aget_type(attr_id);
			if(H5Tget_class(type_id)!=hdfclass){
				HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+" , datatype type mismatch!");
			}
			if(rigid_checks){
				if(H5Tget_size(type_id)!=size){
					HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype size mismatch!");
				}
				if( hdfclass==H5T_INTEGER ){
					if(sign){
						if(H5Tget_sign(type_id)!=H5T_SGN_2){
							HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype sign mismatch!");
						}
					}
					else{
						if(H5Tget_sign(type_id)!=H5T_SGN_NONE){
							HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype sign mismatch!");
						}
					}
				}
			}

			//read
			hid_t errhandle=H5Aread(attr_id,type_id,reinterpret_cast<void*>(&datum));
			errhandle=H5Aclose(attr_id);
			errhandle=H5Tclose(type_id);
		}

		template<typename ctype>
		void rdAtt(const std::string& obj_name, const std::string& attr_name, multi1d<ctype>& datum, const H5T_class_t& hdfclass, const bool& sign, const bool& rigid_checks=true){
			//get datatype properties:
			unsigned int size=sizeof(ctype);

			std::string oname(obj_name), aname(attr_name);
			bool exists=objectExists(current_group,oname);
			if(!exists){
				HDF5_error_exit("HDF5Reader::readAttribute: error, object "+oname+" you try to read attribute from does not exists!");
			}
      
			//do sanity checks and get datatype
			hid_t ex=H5Aexists_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			if(ex!=1){
				HDF5_error_exit("HDF5Reader::readAttribute: error, the attribute "+aname+" you try to read does not exists!");
			}
			hid_t attr_id=H5Aopen_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT,H5P_DEFAULT);
			if(attr_id<0){
				HDF5_error_exit("HDF5Reader::readAttribute: error, cannot open attribute "+aname+" attached to "+oname+"!");
			}
			hid_t type_id=H5Aget_type(attr_id);
			if(H5Tget_class(type_id)!=hdfclass){
				HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype mismatch!");
			}
			if(rigid_checks){
				if(H5Tget_size(type_id)!=size){
					HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype size mismatch!");
				}
				if( hdfclass==H5T_INTEGER ){
					if(sign){
						if(H5Tget_sign(type_id)!=H5T_SGN_2){
							HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype sign mismatch!");
						}
					} 
					else{
						if(H5Tget_sign(type_id)!=H5T_SGN_NONE){
							HDF5_error_exit("HDF5Reader::readAttribute: error reading "+attr_name+", datatype sign mismatch!");
						}
					}
				}
			}

			//read
			hid_t space_id=H5Aget_space(attr_id);
			if(space_id<0){
				HDF5_error_exit("HDF5Reader::readAttribute: cannot open dataspace.");
			}
			hsize_t space_size=H5Sget_simple_extent_npoints(space_id);
			H5Sclose(space_id);
			ctype* token=new ctype[space_size];
			herr_t errhandle=H5Aread(attr_id,type_id,reinterpret_cast<void*>(token));
			H5Aclose(attr_id);
			H5Tclose(type_id);
			datum.resize(space_size);
			for(hsize_t i=0; i<space_size; i++) datum[i]=token[i];
			delete [] token;
		}

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING DATASETS HELPERS                                                                                                           
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		template<typename ctype>
		void rd(const std::string& dataname, ctype& datum, const H5T_class_t& hdfclass, const bool& sign, const bool& rigid_checks=true){
			//get datatype properties:
			unsigned int size=sizeof(ctype);

			std::string dname(dataname);
			bool exists=objectExists(current_group,dname);
			if(!exists){
				HDF5_error_exit("HDF5Reader::read: error, dataset does not exists!");
			}
			H5O_info_t objinfo;
			herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
			if(objinfo.type!=H5O_TYPE_DATASET){
				HDF5_error_exit("HDF5Reader::read: error, "+dname+" exists but it is not a dataset!");
			}
			hid_t dset_id=H5Dopen(current_group,dname.c_str(),H5P_DEFAULT);
			if(dset_id<0){
				HDF5_error_exit("HDF5Reader::read: error reading "+dataname+", cannot open dataset!");
			}
			hid_t type_id=H5Dget_type(dset_id);
			if(H5Tget_class(type_id)!=hdfclass){
				HDF5_error_exit("HDF5Reader::read: error reading "+dataname+", datatype mismatch!");
			}
			//do sanity checks and get datatype 
			if(rigid_checks){
				if(H5Tget_size(type_id)!=size){
					HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype size mismatch!");
				}
				if( hdfclass==H5T_INTEGER ){
					if(sign){
						if(H5Tget_sign(type_id)!=H5T_SGN_2){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					} 
					else{
						if(H5Tget_sign(type_id)!=H5T_SGN_NONE){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					}
				}
			}

			//read
			hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dread(dset_id,type_id,H5S_ALL,H5S_ALL,plist_id,static_cast<void*>(&datum));
			H5Pclose(plist_id);
			H5Dclose(dset_id);
			H5Tclose(type_id);
		}

		template<typename ctype>
		void rd(const std::string& dataname, multi1d<ctype>& datum, const H5T_class_t& hdfclass, const bool& sign, const bool& rigid_checks=true){
			//get datatype properties:
			unsigned int size=sizeof(ctype);

			std::string dname(dataname);
			bool exists=objectExists(current_group,dname);
			if(!exists){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", dataset does not exists!");
			}
			H5O_info_t objinfo;
			herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
			if(objinfo.type!=H5O_TYPE_DATASET){
				HDF5_error_exit("HDF5::read: error, "+dname+" exists but it is not a dataset!");
			}
			hid_t dset_id=H5Dopen(current_group,dname.c_str(),H5P_DEFAULT);
			if(dset_id<0){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", cannot open dataset!");
			}
			hid_t type_id=H5Dget_type(dset_id);
			if(H5Tget_class(type_id)!=hdfclass){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype mismatch!");
			}
			//do sanity checks and get datatype 
			if(rigid_checks){
				if(H5Tget_size(type_id)!=size){
					HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype size mismatch!");
				}
				if( hdfclass==H5T_INTEGER ){
					if(sign){
						if(H5Tget_sign(type_id)!=H5T_SGN_2){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					} 
					else{
						if(H5Tget_sign(type_id)!=H5T_SGN_NONE){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					}
				}
			}

			//read
			hid_t space_id=H5Dget_space(dset_id);
			if(space_id<0){
				HDF5_error_exit("HDF5::read: error, the dataset is corrupted!");
			}
			//check for correct dimensionality
			hsize_t dim=H5Sget_simple_extent_ndims(space_id);
			if(dim!=1){
				HDF5_error_exit("HDF5::read: error, dimension mismatch!");
			}
			hsize_t space_size=H5Sget_simple_extent_npoints(space_id);
			H5Sclose(space_id);
			ctype* token=new ctype[space_size];
			hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dread(dset_id,type_id,H5S_ALL,H5S_ALL,plist_id,static_cast<void*>(token));
			H5Pclose(plist_id);
			H5Dclose(dset_id);
			H5Tclose(type_id);
			datum.resize(space_size);
			for(hsize_t i=0; i<space_size; i++) datum[i]=token[i];
			delete [] token;
		}

		template<typename ctype>
		void rd(const std::string& dataname, multi2d<ctype>& datum, const H5T_class_t& hdfclass, const bool& sign, const bool& rigid_checks=true){
			//get datatype properties:
			unsigned int size=sizeof(ctype);

			std::string dname(dataname);
			bool exists=objectExists(current_group,dname);
			if(!exists){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", dataset does not exists!");
			}
			H5O_info_t objinfo;
			herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
			if(objinfo.type!=H5O_TYPE_DATASET){
				HDF5_error_exit("HDF5::read: error, "+dname+" exists but it is not a dataset!");
			}
			hid_t dset_id=H5Dopen(current_group,dname.c_str(),H5P_DEFAULT);
			if(dset_id<0){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", cannot open dataset!");
			}
			hid_t type_id=H5Dget_type(dset_id);
			if(H5Tget_class(type_id)!=hdfclass){
				HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype mismatch!");
			}
			//do sanity checks and get datatype 
			if(rigid_checks){
				if(H5Tget_size(type_id)!=size){
					HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype size mismatch!");
				}
				if( hdfclass==H5T_INTEGER ){
					if(sign){
						if(H5Tget_sign(type_id)!=H5T_SGN_2){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					} 
					else{
						if(H5Tget_sign(type_id)!=H5T_SGN_NONE){
							HDF5_error_exit("HDF5::read: error reading "+dataname+", datatype sign mismatch!");
						}
					}
				}
			}

			//read
			hid_t space_id=H5Dget_space(dset_id);
			if(space_id<0){
				HDF5_error_exit("HDF5::read: error, the dataset is corrupted!");
			}
			hsize_t dim=H5Sget_simple_extent_ndims(space_id);
			if(dim!=2){
				HDF5_error_exit("HDF5::read: error, dimension mismatch!");
			}
			hsize_t dims[2];
			errhandle=H5Sget_simple_extent_dims(space_id, dims, NULL);
			hsize_t space_size=H5Sget_simple_extent_npoints(space_id);
			H5Sclose(space_id);
			ctype* token=new ctype[space_size];
			hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dread(dset_id,type_id,H5S_ALL,H5S_ALL,plist_id,static_cast<void*>(token));
			H5Pclose(plist_id);
			H5Dclose(dset_id);
			H5Tclose(type_id);
			datum.resize(dims[0],dims[1]);
			for(hsize_t i=0; i<dims[0]; i++){
				for(hsize_t j=0; j<dims[1]; j++){
					datum(i,j)=token[j+dims[1]*i]; //HDF5 stores row-major
				}
			}
			delete [] token;
		}

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING CUSTOM OBJECTS HELPERS                                                                                                     
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//helper routines for reading and writing objects:
		void readPrepare(const std::string& name, hid_t& type_id);
		void readPrepare(const std::string& name, hid_t& type_id, multi1d<ullong>& sizes);

		void readLattice(const std::string& name, const hid_t& type_id, const hid_t& base_type_id,
		const ullong& obj_size, const ullong& tot_size, REAL* buf);

	public:
		//open and close files. Open is virtual since the openmode differs for reader and writer:                                                        
		virtual void open(const std::string& filename) = 0;
		int close();

		//find out if file exists:
		bool check_exists(const std::string& filename)const;

		//setting the stripesize:
		void set_stripesize(const int& stripesizee){stripesize=stripesizee;};
    
		//activate profiling
		void set_profiling(const bool& profilee){profile=profilee;};
	
		//print present working directory and parent directory:                                                                                          
		std::string pwd()const;
		std::string parentDir()const;
    
		//step back one directory:
		void pop();
		//change to a directory specified by dirname:
		void cd(const std::string& dirname);

		//reading routines:
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING ATTRIBUTES                                                                                                                 
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		void readAttribute(const std::string& obj_name, const std::string& attr_name, short& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, unsigned short& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, int& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, unsigned int& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, unsigned long long& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, float& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, double& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, std::string& datum);

		//array value
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<short>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<unsigned short>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<int>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<unsigned int>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<unsigned long long>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<float>& datum);
		void readAttribute(const std::string& obj_name, const std::string& attr_name, multi1d<double>& datum);

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING DATASETS                                                                                                                   
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		void read(const std::string& obj_name, short& datum);
		void read(const std::string& obj_name, unsigned short& datum);
		void read(const std::string& obj_name, int& datum);
		void read(const std::string& obj_name, unsigned int& datum);
		void read(const std::string& obj_name, unsigned long long& datum);
		void read(const std::string& obj_name, float& datum);
		void read(const std::string& obj_name, double& datum);
		void read(const std::string& dataname, std::string& datum);

		//array value
		//1D
		void read(const std::string& obj_name, multi1d<short>& datum);
		void read(const std::string& obj_name, multi1d<unsigned short>& datum);
		void read(const std::string& obj_name, multi1d<int>& datum);
		void read(const std::string& obj_name, multi1d<unsigned int>& datum);
		void read(const std::string& obj_name, multi1d<unsigned long long>& datum);
		void read(const std::string& obj_name, multi1d<float>& datum);
		void read(const std::string& obj_name, multi1d<double>& datum);
		//2D
		void read(const std::string& obj_name, multi2d<short>& datum);
		void read(const std::string& obj_name, multi2d<unsigned short>& datum);
		void read(const std::string& obj_name, multi2d<int>& datum);
		void read(const std::string& obj_name, multi2d<unsigned int>& datum);
		void read(const std::string& obj_name, multi2d<unsigned long long>& datum);
		void read(const std::string& obj_name, multi2d<float>& datum);
		void read(const std::string& obj_name, multi2d<double>& datum);
	
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING OSCALAR OBJECTS                                                                                                            
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		template<class T>
		void read(const std::string& name, OScalar<T>& scalar)
		{
			//read dataset extents:
			multi1d<ullong> sizes;
			ullong obj_size=0;
			hid_t type_id;
			readPrepare(name,type_id,sizes);

			//sanity check:
			ullong float_size=0;
			if(H5Tequal(type_id,H5T_NATIVE_FLOAT)>0){
				float_size=4;
			}
			else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)>0){
				float_size=8;
			}
			else{
				HDF5_error_exit("HDF5Reader::read: error, datatype mismatch!\n");
			}
			H5Tclose(type_id);
			if(sizes.size()!=1){
				HDF5_error_exit("HDF5Reader::read: error, wrong dimensionality!");
			}
			obj_size=sizes[0];
			if( (obj_size*float_size) != sizeof(T) ){
				HDF5_error_exit("HDF5Reader::read: error size of input vectors differ from those in record!\n");
			}

			//read data:
			multi1d<REAL> buf(obj_size);
			if(float_size==4){
				multi1d<float> buf32(obj_size);
				this->read(name,buf32);
				for(ullong i=0; i<obj_size; i++) buf[i]=static_cast<REAL>(buf32[i]);
			}
			else{
				multi1d<double> buf64(obj_size);
				this->read(name,buf64);
				for(ullong i=0; i<obj_size; i++) buf[i]=static_cast<REAL>(buf64[i]);
			}
			memcpy(reinterpret_cast<void*>(&scalar.elem()),&buf[0],sizeof(T));
		}
    
		//array
		template<class T>
		void read(const std::string& name, multi1d< OScalar<T> >& scalararray)
		{
			//read dataset extents:
			multi1d<ullong> sizes;
			ullong obj_size=0;
			hid_t type_id;
			readPrepare(name,type_id,sizes);

			//sanity check:
			ullong float_size=0;
			if(H5Tequal(type_id,H5T_NATIVE_FLOAT)>0){
				float_size=4;
			}
			else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)>0){
				float_size=8;
			}
			else{
				HDF5_error_exit("HDF5Reader::read: error, datatype mismatch!\n");
			}
			H5Tclose(type_id);
			if(sizes.size()!=1){
				HDF5_error_exit("HDF5Reader::read: error, wrong dimensionality!");
				return;
			}
			obj_size=sizes[0];
			if( (obj_size*float_size)%sizeof(T) != 0 ){
				HDF5_error_exit("HDF5Reader::read: error size of input vectors differ from those in record!\n");
			}
			ullong arr_size=(obj_size*float_size)/sizeof(T);
			obj_size/=arr_size;

			//read data:
			multi1d<REAL> buf(obj_size*arr_size);
			if(float_size==4){
				multi1d<float> buf32(obj_size*arr_size);
				this->read(name,buf32);
				for(ullong i=0; i<(obj_size*arr_size); i++) buf[i]=static_cast<REAL>(buf32[i]);
			}
			else{
				multi1d<double> buf64(obj_size*arr_size);
				this->read(name,buf64);
				for(ullong i=0; i<(obj_size*arr_size); i++) buf[i]=static_cast<REAL>(buf64[i]);
			}
			scalararray.resize(arr_size);
			for(ullong i=0; i<arr_size; i++){
				memcpy(reinterpret_cast<void*>(&scalararray[i].elem()),&buf[i*obj_size],sizeof(T));
			}
		}
    
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//READING OLATTICE OBJECTS                                                                                                           
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//read OLattice object:
		template<class T>
		void read(const std::string& name, OLattice<T>& field)
		{
			StopWatch swatch_datatypes, swatch_prepare, swatch_reorder, swatch_read;
			
			//read dataset extents:
			if(profile) swatch_prepare.start();
			multi1d<ullong> sizes;
			ullong obj_size=0;
			hid_t type_id;
			readPrepare(name,type_id,sizes);
			if(profile) swatch_prepare.stop();

			//sanity check:
			if(profile) swatch_datatypes.start();
			ullong float_size=0;
			if(H5Tequal(type_id,H5T_NATIVE_FLOAT)>0){
				float_size=4;
			}
			else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)>0){
				float_size=8;
			}
			else{
				HDF5_error_exit("HDF5Reader::read: error, datatype mismatch!");
			}
			if(sizes.size()!=(Nd+1)){
				HDF5_error_exit("HDF5Reader::read: error, wrong dimensionality!");
			}
			for(unsigned int dd=0; dd<Nd; dd++){
				if(sizes[Nd-dd-1]!=Layout::lattSize()[dd]){
					HDF5_error_exit("HDF5Reader::read: mismatching lattice extents.");
				}
			}
			obj_size=sizes[Nd];
			if( (obj_size*float_size) != sizeof(T) ){
				HDF5_error_exit("HDF5Reader::read: error size of input vectors differ from those in record!");
			}
			if(profile) swatch_datatypes.stop();

			//determine local sizes, allocate memory and read
			if(profile) swatch_read.start();
			const int mynode=Layout::nodeNumber();
			const int nodeSites = Layout::sitesOnNode();
			size_t tot_size = obj_size*nodeSites;
			REAL* buf = new(nothrow) REAL[tot_size];
			if( buf == 0x0 ) {
				HDF5_error_exit("Unable to allocate buf\n");
			}
			readLattice(name,type_id,type_id,obj_size,tot_size,buf);
			H5Tclose(type_id);
			if(profile) swatch_read.stop();

			//put lattice into u-field and reconstruct as well as reorder them on the fly:
			// Reconstruct the gauge field
			if(profile) swatch_reorder.start();
			int run=0;
			for(int site=0; site < Layout::vol(); ++site){
				multi1d<int> coord = crtesn(site, Layout::lattSize());

				int node   = Layout::nodeNumber(coord);
				int linear = Layout::linearSiteIndex(coord);

				if (mynode == node){
					memcpy(&(field.elem(linear)),reinterpret_cast<char*>(buf+run*obj_size),float_size*obj_size);
					run++;
				}
			}
			delete [] buf;
			if(profile) swatch_reorder.stop();
			
			if(profile){
				QDPIO::cout << "HDF5-I/O statistics. Read:" << std::endl;
				QDPIO::cout << "\t preparing: " << swatch_prepare.getTimeInSeconds() << std::endl;
				QDPIO::cout << "\t datatype-handling: " << swatch_datatypes.getTimeInSeconds() << std::endl;
				QDPIO::cout << "\t reordering: " << swatch_reorder.getTimeInSeconds() << std::endl;
				QDPIO::cout << "\t read: " << swatch_read.getTimeInSeconds() << std::endl;
				QDPIO::cout << "\t MB read: " << Layout::vol()*sizeof(T)/1024/1024 << std::endl;
			}
		}

		//read multi1d<OLattice> object: 
		template<class T>
		void read(const std::string& name, multi1d< OLattice<T> >& fieldarray)
		{
			StopWatch swatch_datatypes, swatch_prepare, swatch_reorder, swatch_read;
			
			//read dataset extents:
			if(profile) swatch_prepare.start();
			multi1d<ullong> sizes;
			ullong obj_size=0;
			hid_t type_id;
			readPrepare(name,type_id,sizes);
			swatch_prepare.stop();

			//check sanity
			if(profile) swatch_datatypes.start();
			ullong float_size=0;
			if(H5Tequal(type_id,H5T_NATIVE_FLOAT)>0){
				float_size=4;
			}
			else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)>0){
				float_size=8;
			}
			else{
				HDF5_error_exit("HDF5Reader::read: error, datatype mismatch!");
			}

			if(sizes.size()!=(Nd+1)){
				HDF5_error_exit("HDF5Reader::read: error, wrong dimensionality!");
			}
			for(unsigned int dd=0; dd<Nd; dd++){
				if(sizes[Nd-dd-1]!=Layout::lattSize()[dd]){
					HDF5_error_exit("HDF5Reader::read: mismatching lattice extents.");
				}
			}
			obj_size=sizes[Nd];
			if( (obj_size*float_size)%sizeof(T) != 0 ){
				HDF5_error_exit("HDF5Reader::read: error size of input vectors differ from those in record!");
			}
			ullong arr_size=(obj_size*float_size)/sizeof(T);
			obj_size/=arr_size;
			if(profile) swatch_datatypes.stop();

			//determine local sizes, allocate memory and read
			if(profile) swatch_read.start();
			const int mynode=Layout::nodeNumber();
			const int nodeSites = Layout::sitesOnNode();
			size_t tot_size = obj_size*arr_size*nodeSites;
			REAL* buf = new(nothrow) REAL[tot_size];
			if( buf == 0x0 ) {
				HDF5_error_exit("Unable to allocate buf!");
			}
			readLattice(name,type_id,type_id,obj_size*arr_size,tot_size,buf);
			H5Tclose(type_id);
			if(profile) swatch_read.stop();

			//put lattice into u-field and reconstruct as well as reorder them on the fly:
			// Reconstruct the gauge field
			if(profile) swatch_reorder.start();
			fieldarray.resize(arr_size);
			int run=0;
			for(int site=0; site < Layout::vol(); ++site){
				multi1d<int> coord = crtesn(site, Layout::lattSize());

				int node   = Layout::nodeNumber(coord);
				int linear = Layout::linearSiteIndex(coord);

				if (mynode == node){
					for(unsigned int dd=0; dd<arr_size; dd++){
						memcpy(&(fieldarray[dd].elem(linear)),reinterpret_cast<char*>(buf+run*obj_size),float_size*obj_size);
						run++;
					}
				}
			}
			delete [] buf;
			if(profile) swatch_reorder.stop();
			
			if(profile){
				QDPIO::cout << "HDF5-I/O statistics. Read:" << std::endl;
				QDPIO::cout << "\t preparing: " << swatch_prepare.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t datatype-handling: " << swatch_datatypes.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t reordering: " << swatch_reorder.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t read: " << swatch_read.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t MB read: " << Layout::vol()*fieldarray.size()*sizeof(T)/1024/1024 << std::endl;
			}
		}

		//special gauge file formats
		void readFUEL(const std::string& name, multi1d<LatticeColorMatrixD3>& field);
    
	};

	//template specializations:
	//complex types
	//single datum
	template<>void HDF5::read< PScalar< PScalar< RComplex<float> > > >(const std::string& dataname, ComplexF& datum);
	template<>void HDF5::read< PScalar< PScalar< RComplex<double> > > >(const std::string& dataname, ComplexD& datum);

	//array value
	template<>void HDF5::read< PScalar< PScalar< RComplex<float> > > >(const std::string& dataname, multi1d<ComplexF>& datum);
	template<>void HDF5::read< PScalar< PScalar< RComplex<double> > > >(const std::string& dataname, multi1d<ComplexD>& datum);

	//specializations for Lattice objects
	template<>void HDF5::read< PScalar< PColorMatrix< RComplex<REAL64>, 3> > >(const std::string& name, LatticeColorMatrixD3& field);

	//specializations for multi1d<OLattice> objects
	template<>void HDF5::read< PScalar< PColorMatrix< RComplex<REAL64>, 3> > >(const std::string& name, multi1d<LatticeColorMatrixD3>& field);
	//--------------------------------------------------------------------------------
	//! HDF5 reader class
	/*!
	This is used to read data from an HDF5 file.
	*/
	class HDF5Reader : public HDF5
	{      
	public:
		//! Empty constructor
		HDF5Reader();
		HDF5Reader(const long int& stripesize, const long int& maxalign=0);
      
		//! Construct from contents of file
		/*! Opens and reads an XML file.
		\param filename The name of the file.
		*/
		HDF5Reader(const std::string& filename);
      
		//! Destructor:
		~HDF5Reader();

		//open file and other useful stuff:                                                                                                                                     
		void open(const std::string& filename);

	};
  
	//--------------------------------------------------------------------------------                                                                 
	//! HDF5 writer class                                                                                                                              
	/*!                                                                                                                                                
	This is used to write data to an HDF5 file.                                                                                               
	*/
	class HDF5Writer : public HDF5
	{
	private:
		//helpers for committing datatypes:
		void commitType(const std::string& name, hid_t dtype_id){
			//first, check if type is already committed:
			htri_t iscommitted=H5Tcommitted(dtype_id);
			if(iscommitted==0){
				//not commit, commit type:
				herr_t errhandle=H5Tcommit(file_id,name.c_str(),dtype_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			}
		}

		//helpers for attribute handling
		static herr_t rmAtt(hid_t location_id, const char *attr_name, const H5A_info_t* attrinfo, void* opdata);

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING ATTRIBUTES HELPERS
		//***********************************************************************************************************************************
		//*********************************************************************************************************************************** 
		template<typename ctype>
		void wtAtt(const std::string& obj_name, const std::string& attr_name, const ctype& datum, const hid_t& hdftype, const bool& overwrite){
			std::string oname(obj_name), aname(attr_name);

			bool exists=objectExists(current_group,oname);
			if(!exists){
				QDPIO::cout << "HDF5Writer::writeAttribute: error, object " << oname << " you try to write attribute to does not exists!" << std::endl;
				return;
			}

			hid_t ex=H5Aexists_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			if(ex==1){
				if(!overwrite){
					QDPIO::cout << "HDF5Writer::writeAttribute: error, attribute " << aname << " already exists!" << std::endl;
					return;
				}
				herr_t errhandle=H5Adelete_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			}

			ctype datumcpy=datum;
			hid_t attr_space_id=H5Screate(H5S_SCALAR);
			hid_t attr_id=H5Acreate_by_name(current_group,oname.c_str(),aname.c_str(),hdftype,attr_space_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			herr_t errhandle=H5Awrite(attr_id,hdftype,reinterpret_cast<void*>(&datumcpy));
			H5Aclose(attr_id);
			H5Sclose(attr_space_id);
		}

		template<typename ctype>
		void wtAtt(const std::string& obj_name, const std::string& attr_name, const multi1d<ctype>& datum, const hid_t& hdftype, const bool& overwrite){
			std::string oname(obj_name), aname(attr_name);

			bool exists=objectExists(current_group,oname);
			if(!exists){
				QDPIO::cout << "HDF5Writer::writeAttribute: error, object " << oname << " you try to write attribute to does not exists!" << std::endl;
				return;
			}

			hid_t ex=H5Aexists_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			if(ex==1){
				if(!overwrite){
					QDPIO::cout << "HDF5Writer::writeAttribute: error, attribute " << aname << " already exists!" << std::endl;
					return;
				}
				herr_t errhandle=H5Adelete_by_name(current_group,oname.c_str(),aname.c_str(),H5P_DEFAULT);
			}

			hsize_t dimcount=datum.size();
			ctype* tmpdim=new ctype[dimcount];
			for(unsigned int i=0; i<dimcount; i++) tmpdim[i]=datum[i];
			hid_t attr_space_id=H5Screate_simple(1,const_cast<const hsize_t*>(&dimcount),const_cast<const hsize_t*>(&dimcount));
			hid_t attr_id=H5Acreate_by_name(current_group,oname.c_str(),aname.c_str(),hdftype,attr_space_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			hid_t errhandle=H5Awrite(attr_id,hdftype,reinterpret_cast<void*>(tmpdim));
			delete [] tmpdim;
			H5Aclose(attr_id);
			H5Sclose(attr_space_id);
		}  

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING DATASETS HELPERS                                                                                                         
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************  
		template<typename ctype>
		void wt(const std::string& dataname, const ctype& datum, const hid_t& hdftype, const bool& overwrite){ 
			std::string dname(dataname);
			hid_t dataid, spaceid;
      
			bool exists=objectExists(current_group,dname);
			if(exists){
				if(!overwrite){
					QDPIO::cout << "HDF5Writer::write: error, dataset already exists!" << std::endl;
					return;
				}
				H5O_info_t objinfo;
				herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
				if(objinfo.type!=H5O_TYPE_DATASET){
					QDPIO::cout << "HDF5Writer::write: error, object you try to write does already exist and is of different type!" << std::endl;
					return;
				}
				errhandle=H5Ldelete(current_group,dname.c_str(),H5P_DEFAULT);
			}
       
			spaceid=H5Screate(H5S_SCALAR);
			dataid=H5Dcreate(current_group,dname.c_str(),hdftype,spaceid,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			H5Sclose(spaceid);

			ctype datumcpy=datum;
			hid_t plist_id = H5Pcreate (H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dwrite(dataid,hdftype,H5S_ALL,H5S_ALL,plist_id,reinterpret_cast<void*>(&datumcpy));
			H5Pclose(plist_id);
			H5Dclose(dataid);
		}

		template<typename ctype>
		void wt(const std::string& dataname, const multi1d<ctype>& datum, const hid_t& hdftype, const bool& overwrite){
			std::string dname(dataname);
			hid_t dataid, spaceid;

			bool exists=objectExists(current_group,dname);
			if(exists){
				if(!overwrite){
					QDPIO::cout << "HDF5Writer::write: error, object named " << dname << " already exists!" << std::endl;
					return;
				}
				H5O_info_t objinfo;
				herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
				if(objinfo.type!=H5O_TYPE_DATASET){
					QDPIO::cout << "HDF5Writer::write: error, object you try to write does already exist and is of different type!" << std::endl;
					return;
				}
				errhandle=H5Ldelete(current_group,dname.c_str(),H5P_DEFAULT);
			}
       
			spaceid=H5Screate(H5S_SIMPLE);
			hsize_t size[1];
			size[0]=static_cast<hsize_t>(datum.size());
			herr_t errhandle=H5Sset_extent_simple(spaceid,1,size,size);
			dataid=H5Dcreate(current_group,dname.c_str(),hdftype,spaceid,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
			H5Sclose(spaceid);

			ctype* datumcpy=new ctype[datum.size()];
			for(ullong i=0; i<datum.size(); i++) datumcpy[i]=datum[i];
			hid_t plist_id = H5Pcreate (H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dwrite(dataid,hdftype,H5S_ALL,H5S_ALL,plist_id,static_cast<void*>(datumcpy));
			H5Pclose(plist_id);
			H5Dclose(dataid);
			delete [] datumcpy;
		}
		
		template<typename ctype>
		void wt(const std::string& dataname, const multi2d<ctype>& datum, const hid_t& hdftype, const bool& overwrite){
			std::string dname(dataname);
			hid_t dataid, spaceid;

			bool exists=objectExists(current_group,dname);
			if(exists){
				if(!overwrite){
					QDPIO::cout << "HDF5Writer::write: error, object named " << dname << " already exists!" << std::endl;
					return;
				}
				H5O_info_t objinfo;
				herr_t errhandle=H5Oget_info_by_name(current_group,dname.c_str(),&objinfo,H5P_DEFAULT);
				if(objinfo.type!=H5O_TYPE_DATASET){
					QDPIO::cout << "HDF5Writer::write: error, object you try to write does already exist and is of different type!" << std::endl;
					return;
				}
				errhandle=H5Ldelete(current_group,dname.c_str(),H5P_DEFAULT);
			}
       
			//create dataspace and dataset: 
			hsize_t rank = static_cast<hsize_t>(2);
			hsize_t spacesize[2];
			//twist in how HDF5 handles data description
			spacesize[1]=datum.size1();
			spacesize[0]=datum.size2();
			spaceid = H5Screate_simple(rank, const_cast<const hsize_t*>(spacesize), NULL);
			hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
	   		dataid=H5Dcreate(current_group,dname.c_str(),hdftype,spaceid,H5P_DEFAULT,dcpl_id,H5P_DEFAULT);
			H5Pclose(dcpl_id);
			H5Sclose(spaceid);

			ctype* datumcpy=new ctype[spacesize[0]*spacesize[1]];
			for(ullong i=0; i<spacesize[0]; i++){
				for(ullong j=0; j<spacesize[1]; j++){
					datumcpy[j+spacesize[1]*i]=datum(i,j);	//row-major
				}
			}
			hid_t plist_id = H5Pcreate (H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			H5Dwrite(dataid,hdftype,H5S_ALL,H5S_ALL,plist_id,static_cast<void*>(datumcpy));
			H5Pclose(plist_id);
			H5Dclose(dataid);
			delete [] datumcpy;
		}
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING CUSTOM OBJECTS HELPERS                                                                                                                                                                      
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//helper routines for Lattice field I/O:
		void writePrepare(const std::string& name, const bool& overwrite);
		void writeLattice(const std::string& name, const hid_t& datatype, const ullong& obj_size, char* buf);

	public:
		//! Empty constructors
		HDF5Writer();
		HDF5Writer(const long int& stripesize, const long int& maxalign=0);

		//! Construct from contents of file
		HDF5Writer(const std::string& filename, const bool& overwrite=false);

		//! Destructor
		~HDF5Writer();

		//open file:
		void open(const std::string& filename){
			open(filename,false);
		}

		void open(const std::string& filename, const bool& overwrite);

		/*!
		Creates a new group and steps down into it. Creates new groups on the way down the tree:
		*/
		void push(const std::string& name);

		//delete all attributes attached to a dataset with name "obj_name"
		void deleteAllAttributes(const std::string& obj_name);
		void deleteAttribute(const std::string& obj_name, const std::string& attr_name);

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING ATTRIBUTES
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const short& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const unsigned short& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const int& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const unsigned int& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const unsigned long long& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const float& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const double& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const std::string& datum, const bool& overwrite=false);
    
		//array value:
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<short>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<unsigned short>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<int>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<unsigned int>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<unsigned long long>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<float>& datum, const bool& overwrite=false);
		void writeAttribute(const std::string& obj_name, const std::string& attr_name, const multi1d<double>& datum, const bool& overwrite=false);
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING DATASETS
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		void write(const std::string& obj_name, const short& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const unsigned short& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const int& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const unsigned int& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const unsigned long long& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const float& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const double& datum, const bool& overwrite=false);
		void write(const std::string& dataname, const std::string& datum, const bool& overwrite=false);

		//array value
		//1D
		void write(const std::string& obj_name, const multi1d<short>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<unsigned short>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<int>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<unsigned int>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<unsigned long long>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<float>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi1d<double>& datum, const bool& overwrite=false);
		//2D
		void write(const std::string& obj_name, const multi2d<short>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<unsigned short>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<int>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<unsigned int>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<unsigned long long>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<float>& datum, const bool& overwrite=false);
		void write(const std::string& obj_name, const multi2d<double>& datum, const bool& overwrite=false);
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING Compound Datatypes
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//single datum
		template<class T>
		void write(const std::string& name, const OScalar<T>& scalar, const bool& overwrite=false)
		{
			//copy buffer into data
			size_t float_size=sizeof(REAL);
			size_t obj_size=sizeof(T)/float_size;
			multi1d<REAL> buf(obj_size);
			memcpy(reinterpret_cast<char*>(&buf[0]),&(scalar.elem()),sizeof(T));
			write(name,buf,overwrite);
		}

		//array:
		template<class T>
		void write(const std::string& name, const multi1d< OScalar<T> >& scalararray, const bool& overwrite=false)
		{
			//copy buffer into data
			size_t float_size=sizeof(REAL);
			size_t obj_size=sizeof(T)/float_size;
			size_t arr_size=scalararray.size();
			multi1d<REAL> buf(obj_size*arr_size);
			for(ullong i=0; i<arr_size; i++){
				memcpy(reinterpret_cast<char*>(&buf[i*obj_size]),&(scalararray[i].elem()),sizeof(T));
			}
			write(name,buf,overwrite);
		}

		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		//WRITING OLATTICE OBJECTS                                                                                                           
		//***********************************************************************************************************************************
		//***********************************************************************************************************************************
		template<class T>
		void write(const std::string& name, const OLattice<T>& field, const bool& overwrite=false)
		{
			StopWatch swatch_prepare, swatch_reorder, swatch_write;
			
			//before writing is performed, check if dataset exists:
			if(profile) swatch_prepare.start();
			writePrepare(name,overwrite);
			if(profile) swatch_prepare.stop();

			//get node information:
			if(profile) swatch_reorder.start();
			const int mynode=Layout::nodeNumber();
			const int nodeSites = Layout::sitesOnNode();

			//copy buffer into data
			size_t float_size=sizeof(REAL);
			size_t obj_size=sizeof(T)/float_size;
			REAL* buf=new REAL[nodeSites*obj_size];
			int run=0;
			for(int site=0; site < Layout::vol(); ++site){
				multi1d<int> coord = crtesn(site, Layout::lattSize());
				int node   = Layout::nodeNumber(coord);

				if(node==mynode){
					int linear = Layout::linearSiteIndex(coord);
					memcpy(reinterpret_cast<char*>(buf+run*obj_size),&(field.elem(linear)),sizeof(T));
					run++;
				}
			}
			if(profile) swatch_reorder.stop();

			//determine datatype:
			hid_t type_id;
			if(float_size==4){
				type_id=H5Tcopy(H5T_NATIVE_FLOAT);
			}
			else if(float_size==8){
				type_id=H5Tcopy(H5T_NATIVE_DOUBLE);
			}
			else{
				HDF5_error_exit("HDF5Writer::write: error, unknown datatype in Lattice IO!");
			}

			//write out the stuff:
			if(profile) swatch_write.start();
			writeLattice(name,type_id,obj_size,reinterpret_cast<char*>(buf));
      
			//clean up
			H5Tclose(type_id);
			delete [] buf;
			if(profile) swatch_write.stop();
			
			if(profile){
				QDPIO::cout << "HDF5-I/O statistics. Write:" << std::endl;
				QDPIO::cout << "\t preparing: " << swatch_prepare.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t reordering: " << swatch_reorder.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t write: " << swatch_write.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t MB written: " << Layout::vol()*sizeof(T)/1024/1024 << std::endl;
			}
		}

		template<class T>
		void write(const std::string& name, const multi1d< OLattice<T> >& fieldarray, const bool& overwrite=false)
		{
			StopWatch swatch_prepare, swatch_reorder, swatch_write;
			
			//before writing is performed, check if dataset exists:
			if(profile) swatch_prepare.start();
			writePrepare(name,overwrite);
			if(profile) swatch_prepare.stop();
	  
			//get node information:
			if(profile) swatch_reorder.start();
			const int mynode=Layout::nodeNumber();
			const int nodeSites = Layout::sitesOnNode();

			//copy buffer into data
			size_t float_size=sizeof(REAL);
			size_t obj_size=sizeof(T)/float_size;
			size_t arr_size=fieldarray.size();
			REAL* buf=new REAL[nodeSites*obj_size*arr_size];
			int run=0;
			for(int site=0; site < Layout::vol(); ++site){
				multi1d<int> coord = crtesn(site, Layout::lattSize());
				int node   = Layout::nodeNumber(coord);

				if(node==mynode){
					for(unsigned int dd=0; dd<fieldarray.size(); dd++){
						int linear = Layout::linearSiteIndex(coord);
						memcpy(reinterpret_cast<char*>(buf+run*obj_size),&(fieldarray[dd].elem(linear)),sizeof(T));
						run++;
					}
				}
			}

			hid_t type_id;
			if(float_size==4){
				type_id=H5Tcopy(H5T_NATIVE_FLOAT);
			}
			else if(float_size==8){
				type_id=H5Tcopy(H5T_NATIVE_DOUBLE);
			}
			else{
				HDF5_error_exit("HDF5Writer::write: error, unknown datatype in Lattice IO!");
			}
			if(profile) swatch_reorder.stop();

			//write out the stuff:
			if(profile) swatch_write.start();
			writeLattice(name,type_id,obj_size*arr_size,reinterpret_cast<char*>(buf));

			//clean up
			H5Tclose(type_id);
			delete [] buf;
			if(profile) swatch_write.stop();
			
			if(profile){
				QDPIO::cout << "HDF5-I/O statistics. Write:" << std::endl;
				QDPIO::cout << "\t preparing: " << swatch_prepare.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t reordering: " << swatch_reorder.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t write: " << swatch_write.getTimeInSeconds() << " s." << std::endl;
				QDPIO::cout << "\t MB written: " << Layout::vol()*fieldarray.size()*sizeof(T)/1024/1024 << std::endl;
			}
		}

		//special gauge archive IO:
		//FUEL:
		void writeFUEL(const std::string& name, const multi1d<LatticeColorMatrixD3>& field, const bool& overwrite=false);

	};

	//template specializations for OScalar<T> datatypes:
	//complex types
	//single datum
	template<>void HDF5Writer::write< PScalar< PScalar< RComplex<float> > > >(const std::string& dataname, const ComplexF& datum, const bool& overwrite);
	template<>void HDF5Writer::write< PScalar< PScalar< RComplex<double> > > >(const std::string& dataname, const ComplexD& datum, const bool& overwrite);

	//array:
	template<>void HDF5Writer::write< PScalar< PScalar< RComplex<float> > > >(const std::string& dataname, const multi1d<ComplexF>& datum, const bool& overwrite);
	template<>void HDF5Writer::write< PScalar< PScalar< RComplex<double> > > >(const std::string& dataname, const multi1d<ComplexD>& datum, const bool& overwrite);

	//ColorMatrix
	//single datum
	template<>void HDF5Writer::write< PScalar< PColorMatrix< RComplex<REAL32>, 3> > >(const std::string& dataname, const ColorMatrixF3& datum, const bool& overwrite);
	template<>void HDF5Writer::write< PScalar< PColorMatrix< RComplex<REAL64>, 3> > >(const std::string& dataname, const ColorMatrixD3& datum, const bool& overwrite);

	//LatticeColorMatrix
	template<>void HDF5Writer::write< PScalar< PColorMatrix< RComplex<REAL32>, 3> > >(const std::string& name, const LatticeColorMatrixF3& field, const bool& overwrite);
	template<>void HDF5Writer::write< PScalar< PColorMatrix< RComplex<REAL64>, 3> > >(const std::string& name, const LatticeColorMatrixD3& field, const bool& overwrite);

	//multi1d<OLattice> specializations
	template<>void HDF5Writer::write< PScalar< PColorMatrix< RComplex<REAL64>, 3> > >(const std::string& name, const multi1d<LatticeColorMatrixD3>& field, const bool& overwrite);

	//FUEL IO:
	void writeFUEL(const std::string& name, const multi1d<LatticeColorMatrixD3>& field, const bool& overwrite=false);
}
#endif
