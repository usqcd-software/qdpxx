// -*- C++ -*-

/*! @file
* @brief Lattice layout
*
* Lattice layout namespace and operations
*/

#ifndef QDP_LAYOUT_H
#define QDP_LAYOUT_H

#include <cstdint>
namespace QDP {

	using index_t = std::uint64_t;

	/*! @defgroup layout  Layout 
	*
	* Namespace holding info on problem size and machine info
	*
	* @{
	*/

	//! Layout namespace holding info on problem size and machine info
	/*! 
	* This is a namespace instead of a class since it is completely static -
	* no such object should be created 
	*
	* The functions here should be common to all architectures
	*/
	namespace Layout
	{
		//! Initialize some fundamental pieces of the layout
		/*! This routine is used to boostrap the   create   function below */
		void init();

		//! Main lattice creation routine
		void create();

		//! Main destruction routine
		void destroy();

		//! Set lattice size -- problem size
		void setLattSize(const multi1d<int>& nrows);

		//! Set SMP flag -- true if using smp/multiprocessor mode on a node
		void setSMPFlag(bool);

		//! Set number of processors in a multi-threaded implementation
		void setNumProc(index_t N);

		//! Returns the logical node number for the corresponding lattice coordinate
		/*! The API requires this function to be here */
		index_t nodeNumber(const multi1d<index_t>& coord) QDP_CONST;

		//returns local lexicographical site coordinate from linear index:
		multi1d<index_t> localLexiCoordFromLinear(const index_t& linearr) QDP_CONST;

		//! The linearized site index within a node for the corresponding lattice coordinate
		/*! The API requires this function to be here */
		index_t linearSiteIndex(const multi1d<index_t>& coord) QDP_CONST;

		//! Reconstruct the lattice coordinate from the node and site number
		/*! 
		* This is the inverse of the nodeNumber and linearSiteIndex functions.
		* The API requires this function to be here.
		*/
		multi1d<index_t> siteCoords(index_t node, index_t index) QDP_CONST;
  
		extern "C" { 
			/* Export this to "C" */
			void QDPXX_getSiteCoords(index_t coord[], index_t node, index_t linear) QDP_CONST;
			index_t QDPXX_getLinearSiteIndex(const index_t coord[]);
			index_t QDPXX_nodeNumber(const index_t coord[]);

		};

		//! Returns the node number of this node
		index_t nodeNumber() QDP_CONST;

		//! Returns the number of nodes
		index_t numNodes() QDP_CONST;
	

		//! Virtual grid (problem grid) lattice size
		const multi1d<index_t>& lattSize() QDP_CONST;

		//! Total lattice volume
		index_t  vol() QDP_CONST;

		//! Number of sites on node
		index_t sitesOnNode() QDP_CONST;

		//! Returns whether this is the primary node
		bool primaryNode() QDP_CONST;

		//! The linearized site index for the corresponding lexicographic site
		index_t linearSiteIndex(index_t lexicosite) QDP_CONST;

		//! Returns the logical node coordinates for this node
		const multi1d<index_t>& nodeCoord() QDP_CONST;

		//! Returns the logical node coordinates for the corresponding lattice coordinate
		multi1d<index_t> nodeCoord(const multi1d<index_t>& coord);

		//! Subgrid (grid on each node) lattice size
		const multi1d<index_t>& subgridLattSize() QDP_CONST;

		//! Returns the logical size of this machine
		const multi1d<index_t>& logicalSize() QDP_CONST;

		//! Returns the node number given some logical node coordinate
		/*! This is not meant to be speedy */
		index_t getNodeNumberFrom(const multi1d<index_t>& node_coord);

		//! Returns the logical node coordinates given some node number
		/*! This is not meant to be speedy */
		multi1d<index_t> getLogicalCoordFrom(index_t node);


		//! Check if I/O grid is defined
		bool isIOGridDefined(void) QDP_CONST;

		//! set the IO node grid
		void setIONodeGrid(const multi1d<index_t>& io_grid);

		//! number of I/O nodes
		index_t numIONodeGrid(void) QDP_CONST;
	
		//! Get the I/O Node grid
		const multi1d<index_t>& getIONodeGrid() QDP_CONST;


	}

	//! Declaration of shift function object
	extern ArrayBiDirectionalMap  shift;


	/*! @} */   // end of group layout

} // namespace QDP

#endif
