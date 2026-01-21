//! Buffer type that automatically returns to the pool on drop.
//!
//! This module provides the [`Buffer`] type, which represents an allocated memory
//! region from the buffer pool. When a `Buffer` is dropped, its memory is
//! automatically returned to the pool for reuse.
//!
//! # Memory Layout
//!
//! The `Buffer` struct is optimized for minimal memory footprint:
//! - `ptr`: 8 bytes (pointer to memory)
//! - `pool`: 8 bytes (Arc reference to pool)
//! - `block`: 8 bytes (pointer to buddy block)
//! - `level`: 1 byte (0-3, only 4 possible values)
//! - `index`: 1 byte (0-63, max index at level 0)
//! - Padding: 6 bytes for alignment
//!
//! Total: 32 bytes per Buffer.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::BufferPool;
use crate::buddy::{BuddyBlock, FreeNode};

/// A buffer allocated from the pool.
///
/// This type provides read/write access to a contiguous memory region. When dropped,
/// the buffer is automatically returned to the pool for reuse.
///
/// The buffer holds an `Arc<BufferPool>` reference to ensure the pool remains valid
/// for the lifetime of the buffer. This guarantees memory safety even if the original
/// pool handle is dropped.
///
/// # Memory Efficiency
///
/// Buffer fields are packed to minimize memory usage:
/// - `level` uses `u8` (only values 0-3 are valid)
/// - `index` uses `u8` (maximum 63 for level 0)
///
/// # Example
///
/// ```rust
/// use ruapc_bufpool::BufferPoolBuilder;
///
/// # fn main() -> std::io::Result<()> {
/// let pool = BufferPoolBuilder::new().build();
/// let mut buffer = pool.allocate(1024 * 1024)?;
///
/// // Write to the buffer
/// buffer[0] = 42;
/// buffer[1] = 43;
///
/// // Read from the buffer
/// assert_eq!(buffer[0], 42);
///
/// // Buffer is returned to the pool when dropped
/// # Ok(())
/// # }
/// ```
pub struct Buffer {
    /// Pointer to the allocated memory.
    ptr: NonNull<u8>,

    /// Reference to the pool for returning the buffer.
    /// This ensures the pool stays alive while any buffer exists.
    pool: Arc<BufferPool>,

    /// Pointer to the buddy block this buffer belongs to.
    block: NonNull<BuddyBlock>,

    /// The allocation level (0-3).
    /// Packed as u8 to minimize struct size.
    level: u8,

    /// Index within the level in the buddy block.
    /// Maximum value is 63 (for level 0), fits in u8.
    index: u8,
}

// SAFETY: Buffer can be sent between threads as it only contains
// raw pointers that are owned by the pool
unsafe impl Send for Buffer {}

// SAFETY: Buffer can be shared between threads as it provides
// exclusive access to its memory region
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Creates a new buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the allocated memory region
    /// * `level` - Allocation level (0-3)
    /// * `index` - Index within the level (0-63 for level 0)
    /// * `block` - Pointer to the owning `BuddyBlock`
    /// * `pool` - Arc reference to the pool for automatic return on drop
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to a valid memory region of the appropriate size for `level`
    /// - The memory will remain valid until the buffer is dropped
    /// - `block` points to a valid `BuddyBlock`
    /// - `level` is in range 0-3
    /// - `index` is valid for the given level
    pub(crate) unsafe fn new(
        ptr: NonNull<u8>,
        level: usize,
        index: usize,
        block: NonNull<BuddyBlock>,
        pool: Arc<BufferPool>,
    ) -> Self {
        debug_assert!(level < crate::buddy::NUM_LEVELS, "level must be 0-3");
        debug_assert!(
            index < crate::buddy::NODES_PER_LEVEL[0],
            "index must be less than 64"
        );
        Self {
            ptr,
            pool,
            block,
            #[allow(clippy::cast_possible_truncation)]
            level: level as u8, // Safe: level is validated to be < NUM_LEVELS
            #[allow(clippy::cast_possible_truncation)]
            index: index as u8, // Safe: index is validated to be < NODES_PER_LEVEL[0]
        }
    }

    /// Returns the length of the buffer in bytes.
    ///
    /// The length is derived from the allocation level:
    /// - Level 0: 1 MiB
    /// - Level 1: 4 MiB
    /// - Level 2: 16 MiB
    /// - Level 3: 64 MiB
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        crate::buddy::LEVEL_SIZES[self.level as usize]
    }

    /// Returns `true` if the buffer is empty.
    ///
    /// Note: Buffers from this pool are never empty (minimum 1 MiB).
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false // Minimum allocation is 1 MiB, never empty
    }

    /// Returns a raw pointer to the buffer's memory.
    #[inline]
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the buffer's memory.
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the buffer as a byte slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Returns the buffer as a mutable byte slice.
    #[inline]
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for len bytes and we have exclusive access
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    /// Returns the allocation level of this buffer (0-3).
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn level(&self) -> usize {
        self.level as usize
    }

    /// Returns the index within the level (0-63).
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn index(&self) -> usize {
        self.index as usize
    }

    /// Returns the buddy block pointer.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn block(&self) -> NonNull<BuddyBlock> {
        self.block
    }

    /// Returns a pointer to the free node for this buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that block pointer is still valid.
    #[allow(dead_code)]
    pub(crate) unsafe fn free_node(&self) -> NonNull<FreeNode> {
        // SAFETY: block is valid and level/index are within bounds
        unsafe {
            (*self.block.as_ptr()).get_free_node_mut(self.level as usize, self.index as usize)
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Return the buffer directly to the pool with O(1) operation.
        // This acquires the pool's mutex lock to perform the deallocation.
        // Since buddy merging is O(1), this is efficient and avoids the
        // latency spikes that could occur with a batched channel approach.
        self.pool
            .return_buffer(self.level as usize, self.index as usize, self.block);
    }
}

impl Deref for Buffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Buffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for Buffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for Buffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len())
            .field("level", &self.level)
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::BufferPoolBuilder;

    #[test]
    fn test_buffer_basic_operations() {
        let pool = BufferPoolBuilder::new().build();
        let mut buffer = pool.allocate(1024).unwrap();

        // Test len
        assert!(buffer.len() >= 1024);
        assert!(!buffer.is_empty());

        // Test write and read
        buffer[0] = 0xAB;
        buffer[1] = 0xCD;
        assert_eq!(buffer[0], 0xAB);
        assert_eq!(buffer[1], 0xCD);

        // Test as_slice
        let slice = buffer.as_slice();
        assert_eq!(slice[0], 0xAB);

        // Test as_mut_slice
        buffer.as_mut_slice()[2] = 0xEF;
        assert_eq!(buffer[2], 0xEF);
    }

    #[test]
    fn test_buffer_deref() {
        let pool = BufferPoolBuilder::new().build();
        let mut buffer = pool.allocate(1024).unwrap();

        // Fill with pattern
        for (i, byte) in buffer.iter_mut().take(100).enumerate() {
            *byte = i as u8;
        }

        // Read back
        for i in 0..100 {
            assert_eq!(buffer[i], i as u8);
        }
    }

    #[test]
    fn test_buffer_debug() {
        let pool = BufferPoolBuilder::new().build();
        let buffer = pool.allocate(1024).unwrap();

        let debug_str = format!("{buffer:?}");
        assert!(debug_str.contains("Buffer"));
        assert!(debug_str.contains("len"));
    }
}
