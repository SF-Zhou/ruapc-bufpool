//! Buffer type that automatically returns to the pool on drop.
//!
//! This module provides the [`Buffer`] type, which represents an allocated memory
//! region from the buffer pool. When a `Buffer` is dropped, its memory is
//! automatically returned to the pool for reuse.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::buddy::{BuddyBlock, FreeNode};

/// Information needed to return a buffer to the pool.
#[derive(Debug)]
pub struct ReturnInfo {
    /// The allocation level (0-3).
    pub level: usize,
    /// Index within the level in the buddy block.
    pub index: usize,
    /// Pointer to the buddy block this buffer belongs to.
    pub block: NonNull<BuddyBlock>,
}

// SAFETY: ReturnInfo only contains raw pointers that are managed by the pool
unsafe impl Send for ReturnInfo {}
unsafe impl Sync for ReturnInfo {}

/// A buffer allocated from the pool.
///
/// This type provides read/write access to a contiguous memory region. When dropped,
/// the buffer is automatically returned to the pool for reuse.
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

    /// Size of the allocated memory.
    len: usize,

    /// The allocation level (0-3).
    level: usize,

    /// Index within the level in the buddy block.
    index: usize,

    /// Pointer to the buddy block this buffer belongs to.
    block: NonNull<BuddyBlock>,

    /// Channel to return the buffer to the pool.
    return_tx: Arc<mpsc::UnboundedSender<ReturnInfo>>,
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
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to a valid memory region of at least `len` bytes
    /// - The memory will remain valid until the buffer is dropped
    /// - `block` points to a valid `BuddyBlock`
    pub(crate) const unsafe fn new(
        ptr: NonNull<u8>,
        len: usize,
        level: usize,
        index: usize,
        block: NonNull<BuddyBlock>,
        return_tx: Arc<mpsc::UnboundedSender<ReturnInfo>>,
    ) -> Self {
        Self {
            ptr,
            len,
            level,
            index,
            block,
            return_tx,
        }
    }

    /// Returns the length of the buffer in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
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
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the buffer as a mutable byte slice.
    #[inline]
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for len bytes and we have exclusive access
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the allocation level of this buffer.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn level(&self) -> usize {
        self.level
    }

    /// Returns the index within the level.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn index(&self) -> usize {
        self.index
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
    /// The caller must ensure the block pointer is still valid.
    #[allow(dead_code)]
    pub(crate) unsafe fn free_node(&self) -> NonNull<FreeNode> {
        // SAFETY: block is valid and level/index are within bounds
        unsafe { (*self.block.as_ptr()).get_free_node_mut(self.level, self.index) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // Return the buffer to the pool via the channel
        // This is non-blocking and works in any context
        let _ = self.return_tx.send(ReturnInfo {
            level: self.level,
            index: self.index,
            block: self.block,
        });
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
            .field("len", &self.len)
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
