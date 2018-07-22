#pragma once

#include <any>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
// TODO could get rid of most headers by using pimpl

// Class is meant as a helper for managing std::shared_ptr lifetimes when they are shared
// with a foreign language through C API.
// Each time when shared_ptr is moved to foreign code it should be done through `addOwnership`
// When foreign code is done with the pointer, `releaseOwnership` should be called.
//
// Storage is thread-safe (internally synchronized with a lock).
//
// Technically there's nothing shared_ptr specific in storage (it uses type-erased any),
// if needed it can be adjusted to work with other kinds of types with similar semantics.
class LifetimeManager
{
	std::mutex mx;
	std::unordered_multimap<void *, std::any> storage; // address => shared_ptr<T>
public:
	LifetimeManager();
	~LifetimeManager();

	template<typename T>
	T *addOwnership(std::shared_ptr<T> ptr)
	{
		auto ret = ptr.get();
		std::unique_lock<std::mutex> lock{ mx };
		storage.emplace(ret, std::move(ptr));
		return ret;
	}
	void releaseOwnership(void *ptr)
	{
		// TODO should separate retrieving any from storage and deleting it
		// deleting can take time and lock is not needed then
		{
			std::unique_lock<std::mutex> lock{ mx };
			if(auto itr = storage.find(ptr); itr != storage.end())
			{
				storage.erase(itr);
				return;
			}
		}

		std::ostringstream out;
		out << "Cannot unregister ownership of pointer " << ptr << " -- was it previously registered?";
		throw std::runtime_error(out.str());
	}

	// TODO reconsider at some stage more explicit global state
	static auto &instance()
	{
		static LifetimeManager manager;
		return manager;
	}
};
