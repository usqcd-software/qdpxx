// -*- C++ -*-
/*! @file
 * @brief Singleton support
 */

#ifndef __qdp_singleton_h__
#define __qdp_singleton_h__

namespace QDP
{

////////////////////////////////////////////////////////////////////////////////
// class template SingletonHolder
// Provides Singleton amenities for a type T
// Please destroy the singleton when you pleased by calling `DestroySingleton`
////////////////////////////////////////////////////////////////////////////////

  template <typename T>
  class SingletonHolder
  {
  public:
    static T& Instance()
    {
      return *getInstance(true);
    }

    static void DestroySingleton()
    {
      if (getInstance(false))
      {
	delete getInstance();
	getInstance() = nullptr;
      }
    }

  private:
    // Helpers
    static T*& getInstance(bool createNew = false)
    {
      static T* instance = nullptr;
      if (!instance && createNew)
	instance = new T;
      return instance;
    }

    // Protection
    SingletonHolder()
    {
    }
  };

} // namespace Chroma


#endif
