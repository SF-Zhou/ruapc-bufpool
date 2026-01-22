//! Memory allocator trait and default implementation.
//!
//! This module provides the [`Allocator`] trait that defines the interface for memory
//! allocation backends, and [`DefaultAllocator`] which uses the standard library's
//! global allocator.

use std::alloc::{Layout, alloc, dealloc};
use std::io::{Error, ErrorKind, Result};

/// Trait for memory allocation backends.
///
/// Implementations of this trait provide the low-level memory allocation and
/// deallocation operations used by the buffer pool. The default implementation
/// uses the standard library's global allocator.
///
/// # Safety
///
/// Implementations must ensure:
/// - `allocate` returns a valid, properly aligned pointer for the requested size
/// - `deallocate` is only called with pointers previously returned by `allocate`
/// - The allocated memory remains valid until `deallocate` is called
///
/// # Example
///
/// ```rust
/// use ruapc_bufpool::Allocator;
/// use std::io::Result;
///
/// struct MyAllocator;
///
/// impl Allocator for MyAllocator {
///     fn allocate(&self, size: usize) -> Result<*mut u8> {
///         // Custom allocation logic
///         # unimplemented!()
///     }
///
///     unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
///         // Custom deallocation logic
///         # unimplemented!()
///     }
/// }
/// ```
pub trait Allocator: Send + Sync {
    /// Allocates memory of the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes to allocate. Must be greater than 0.
    ///
    /// # Returns
    ///
    /// Returns a pointer to the allocated memory on success, or an error if
    /// allocation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation fails due to memory exhaustion or
    /// invalid size.
    fn allocate(&self, size: usize) -> Result<*mut u8>;

    /// Deallocates memory previously allocated by this allocator.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the memory to deallocate. Must have been returned
    ///   by a previous call to `allocate` on this allocator.
    /// * `size` - The size that was passed to the original `allocate` call.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` was returned by a previous call to `allocate` on this allocator
    /// - `size` matches the size passed to the original `allocate` call
    /// - The memory has not already been deallocated
    unsafe fn deallocate(&self, ptr: *mut u8, size: usize);
}

/// Default allocator using the standard library's global allocator.
///
/// This allocator uses `std::alloc::alloc` and `std::alloc::dealloc` for memory
/// management. It aligns allocations to page size for optimal performance with large buffers.
///
/// # Alignment
///
/// - On 64-bit systems: Uses 2MiB alignment for potential huge page support
/// - On other systems: Uses 4KiB page alignment
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultAllocator;

impl DefaultAllocator {
    /// Returns the alignment size for this platform.
    #[cfg(target_pointer_width = "64")]
    const fn alignment() -> usize {
        2 * 1024 * 1024 // 2MiB
    }

    /// Returns the alignment size for this platform.
    #[cfg(not(target_pointer_width = "64"))]
    const fn alignment() -> usize {
        4096 // 4KiB
    }

    /// Creates a new default allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Allocator for DefaultAllocator {
    fn allocate(&self, size: usize) -> Result<*mut u8> {
        if size == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "size must be > 0"));
        }

        // SAFETY: size is non-zero and alignment is a power of 2
        let layout = Layout::from_size_align(size, Self::alignment())
            .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?;

        // SAFETY: layout is valid (non-zero size, valid alignment)
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            Err(Error::new(
                ErrorKind::OutOfMemory,
                "failed to allocate memory",
            ))
        } else {
            Ok(ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        if size == 0 || ptr.is_null() {
            return;
        }

        // SAFETY: size is non-zero and alignment is a power of 2
        if let Ok(layout) = Layout::from_size_align(size, Self::alignment()) {
            // SAFETY: ptr was allocated with this layout by allocate()
            unsafe { dealloc(ptr, layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_allocator_basic() {
        let allocator = DefaultAllocator::new();

        // Allocate 1MiB
        let size = 1024 * 1024;
        let ptr = allocator.allocate(size).unwrap();
        assert!(!ptr.is_null());

        // Write and read back
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, size);
            assert_eq!(*ptr, 0xAB);
            assert_eq!(*ptr.add(size - 1), 0xAB);
        }

        // Deallocate
        unsafe {
            allocator.deallocate(ptr, size);
        }
    }

    #[test]
    fn test_default_allocator_zero_size() {
        let allocator = DefaultAllocator::new();
        let result = allocator.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_allocator_large_allocation() {
        let allocator = DefaultAllocator::new();

        // Allocate 64MiB
        let size = 64 * 1024 * 1024;
        let ptr = allocator.allocate(size).unwrap();
        assert!(!ptr.is_null());

        unsafe {
            allocator.deallocate(ptr, size);
        }
    }
}
