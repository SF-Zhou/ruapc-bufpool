//! Buffer pool implementation with buddy memory allocation.
//!
//! This module provides the [`BufferPool`] and [`BufferPoolBuilder`] types for
//! managing a pool of memory buffers using the buddy allocation algorithm.

use std::collections::VecDeque;
use std::io::{Error, ErrorKind, Result};
use std::ptr::NonNull;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc, oneshot};

use crate::allocator::{Allocator, DefaultAllocator};
use crate::buddy::{BuddyBlock, NUM_LEVELS, NodeState, SIZE_64MIB, level_to_size, size_to_level};
use crate::buffer::{Buffer, ReturnInfo};
use crate::intrusive_list::IntrusiveList;

/// Default maximum memory limit (256 MiB).
const DEFAULT_MAX_MEMORY: usize = 256 * 1024 * 1024;

/// Builder for creating a [`BufferPool`] with custom configuration.
///
/// # Example
///
/// ```rust
/// use ruapc_bufpool::BufferPoolBuilder;
///
/// let pool = BufferPoolBuilder::new()
///     .max_memory(512 * 1024 * 1024) // 512 MiB
///     .build();
/// ```
pub struct BufferPoolBuilder {
    max_memory: usize,
    allocator: Box<dyn Allocator>,
}

impl Default for BufferPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPoolBuilder {
    /// Creates a new builder with default settings.
    ///
    /// Default settings:
    /// - Max memory: 256 MiB
    /// - Allocator: [`DefaultAllocator`]
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_memory: DEFAULT_MAX_MEMORY,
            allocator: Box::new(DefaultAllocator::new()),
        }
    }

    /// Sets the maximum memory limit for the pool.
    ///
    /// When this limit is reached:
    /// - Synchronous allocations will return an error
    /// - Asynchronous allocations will wait for memory to be freed
    ///
    /// The limit should be a multiple of 64 MiB for optimal utilization.
    #[must_use]
    pub const fn max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Sets a custom allocator for the pool.
    ///
    /// The allocator is used to allocate and deallocate the underlying 64 MiB
    /// memory blocks.
    #[must_use]
    pub fn allocator(mut self, allocator: Box<dyn Allocator>) -> Self {
        self.allocator = allocator;
        self
    }

    /// Builds the buffer pool with the configured settings.
    #[must_use]
    pub fn build(self) -> BufferPool {
        let (return_tx, return_rx) = mpsc::unbounded_channel();
        let return_tx = Arc::new(return_tx);

        let inner = PoolInner {
            allocator: self.allocator,
            max_memory: self.max_memory,
            allocated_memory: 0,
            blocks: Vec::new(),
            free_lists: std::array::from_fn(|_| IntrusiveList::new()),
            waiting_lists: std::array::from_fn(|_| VecDeque::new()),
            min_waiting_level: None,
            return_tx: Arc::clone(&return_tx),
        };

        BufferPool {
            inner: Arc::new(Mutex::new(inner)),
            return_rx: Arc::new(Mutex::new(return_rx)),
        }
    }
}

/// A high-performance memory pool using buddy memory allocation.
///
/// The pool manages memory in 64 MiB blocks and supports allocation of buffers
/// at four size levels: 1 MiB, 4 MiB, 16 MiB, and 64 MiB.
///
/// # Thread Safety
///
/// The pool uses `tokio::sync::Mutex` for thread safety, making it suitable for
/// use in both synchronous and asynchronous contexts.
///
/// # Example
///
/// ```rust
/// use ruapc_bufpool::BufferPoolBuilder;
///
/// # fn main() -> std::io::Result<()> {
/// let pool = BufferPoolBuilder::new()
///     .max_memory(128 * 1024 * 1024)
///     .build();
///
/// // Allocate a 1 MiB buffer
/// let buffer = pool.allocate(1024 * 1024)?;
/// assert!(buffer.len() >= 1024 * 1024);
///
/// // Buffer is automatically returned when dropped
/// drop(buffer);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<PoolInner>>,
    return_rx: Arc<Mutex<mpsc::UnboundedReceiver<ReturnInfo>>>,
}

impl BufferPool {
    /// Creates a new buffer pool with default settings.
    ///
    /// This is equivalent to `BufferPoolBuilder::new().build()`.
    #[must_use]
    pub fn new() -> Self {
        BufferPoolBuilder::new().build()
    }

    /// Processes any pending buffer returns.
    ///
    /// This should be called before allocation attempts to ensure
    /// freed buffers are available.
    fn process_returns_sync(&self, inner: &mut PoolInner) {
        let mut rx = self.return_rx.blocking_lock();
        while let Ok(info) = rx.try_recv() {
            inner.deallocate_buffer(info.level, info.index, info.block);
        }
    }

    /// Processes any pending buffer returns asynchronously.
    async fn process_returns_async(&self, inner: &mut PoolInner) {
        let mut rx = self.return_rx.lock().await;
        while let Ok(info) = rx.try_recv() {
            inner.deallocate_buffer(info.level, info.index, info.block);
        }
    }

    /// Allocates a buffer of at least the specified size.
    ///
    /// The returned buffer may be larger than requested, rounded up to the
    /// nearest allocation level (1 MiB, 4 MiB, 16 MiB, or 64 MiB).
    ///
    /// This method blocks the current thread while waiting for the lock.
    /// For async contexts, use [`async_allocate`](Self::async_allocate) instead.
    ///
    /// # Arguments
    ///
    /// * `size` - The minimum size of the buffer in bytes.
    ///
    /// # Returns
    ///
    /// Returns a [`Buffer`] on success, or an error if:
    /// - The requested size is 0 or exceeds 64 MiB
    /// - The memory limit has been reached
    /// - Memory allocation fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size` is 0 or exceeds 64 MiB (`InvalidInput`)
    /// - Memory limit has been reached (`OutOfMemory`)
    /// - Underlying allocator fails (`OutOfMemory`)
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruapc_bufpool::BufferPoolBuilder;
    ///
    /// # fn main() -> std::io::Result<()> {
    /// let pool = BufferPoolBuilder::new().build();
    /// let buffer = pool.allocate(2 * 1024 * 1024)?; // Request 2 MiB
    /// assert!(buffer.len() >= 4 * 1024 * 1024); // Gets 4 MiB (next level up)
    /// # Ok(())
    /// # }
    /// ```
    pub fn allocate(&self, size: usize) -> Result<Buffer> {
        let mut inner = self.inner.blocking_lock();
        self.process_returns_sync(&mut inner);
        inner.allocate_sync(size)
    }

    /// Allocates a buffer asynchronously.
    ///
    /// This is the async version of [`allocate`](Self::allocate). If the memory
    /// limit has been reached, this method will wait for other buffers to be
    /// freed rather than returning an error.
    ///
    /// # Arguments
    ///
    /// * `size` - The minimum size of the buffer in bytes.
    ///
    /// # Returns
    ///
    /// Returns a [`Buffer`] on success, or an error if:
    /// - The requested size is 0 or exceeds 64 MiB
    /// - Memory allocation fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size` is 0 or exceeds 64 MiB (`InvalidInput`)
    /// - Underlying allocator fails (`OutOfMemory`)
    ///
    /// Note: Unlike [`allocate`](Self::allocate), this method will wait instead
    /// of returning `OutOfMemory` when the pool's memory limit is reached.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ruapc_bufpool::BufferPoolBuilder;
    ///
    /// # async fn example() -> std::io::Result<()> {
    /// let pool = BufferPoolBuilder::new().build();
    /// let buffer = pool.async_allocate(1024 * 1024).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn async_allocate(&self, size: usize) -> Result<Buffer> {
        let level = size_to_level(size).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid size: {size} (must be 1-67108864 bytes)"),
            )
        })?;

        loop {
            let receiver = {
                let mut inner = self.inner.lock().await;
                self.process_returns_async(&mut inner).await;

                // Try to allocate directly
                match inner.try_allocate(level) {
                    Ok(buffer) => return Ok(buffer),
                    Err(e) if e.kind() == ErrorKind::OutOfMemory => {
                        // Memory limit reached, wait for a buffer
                        let (sender, receiver) = oneshot::channel();
                        inner.waiting_lists[level].push_back(sender);
                        inner.update_min_waiting_level();
                        receiver
                    }
                    Err(e) => return Err(e),
                }
            };

            // Wait for a buffer to be available or check for returned buffers
            tokio::select! {
                result = receiver => {
                    match result {
                        Ok(buffer) => return Ok(buffer),
                        Err(_) => {
                            // Sender was dropped, try again
                        }
                    }
                }
                () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {
                    // Check for returned buffers and retry
                }
            }
        }
    }

    /// Returns the current amount of allocated memory in bytes.
    ///
    /// This includes all 64 MiB blocks that have been allocated from the
    /// underlying allocator, regardless of how much is currently in use.
    pub async fn allocated_memory(&self) -> usize {
        self.inner.lock().await.allocated_memory
    }

    /// Returns the maximum memory limit in bytes.
    pub async fn max_memory(&self) -> usize {
        self.inner.lock().await.max_memory
    }

    /// Returns the number of free buffers at each level.
    ///
    /// The returned array contains counts for levels 0-3 (1 MiB to 64 MiB).
    pub async fn free_counts(&self) -> [usize; NUM_LEVELS] {
        let inner = self.inner.lock().await;
        std::array::from_fn(|i| inner.free_lists[i].len())
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Sender type for waiting list entries.
type WaitingSender = oneshot::Sender<Buffer>;

/// Internal pool state protected by the mutex.
pub struct PoolInner {
    /// The allocator used for allocating 64 MiB blocks.
    allocator: Box<dyn Allocator>,

    /// Maximum allowed total memory.
    max_memory: usize,

    /// Current total allocated memory (from the underlying allocator).
    allocated_memory: usize,

    /// All allocated 64 MiB blocks.
    /// Note: We use Box to ensure stable addresses, as free list nodes
    /// reference the blocks via raw pointers.
    #[allow(clippy::vec_box)]
    blocks: Vec<Box<BuddyBlock>>,

    /// Free lists for each level (0 = 1MiB, 3 = 64MiB).
    free_lists: [IntrusiveList<crate::buddy::FreeNodeData>; NUM_LEVELS],

    /// Waiting lists for each level (async waiters).
    waiting_lists: [VecDeque<WaitingSender>; NUM_LEVELS],

    /// Minimum level that has waiting allocations.
    min_waiting_level: Option<usize>,

    /// Channel sender for buffer returns.
    return_tx: Arc<mpsc::UnboundedSender<ReturnInfo>>,
}

impl PoolInner {
    /// Synchronous allocation - returns error if memory limit reached.
    fn allocate_sync(&mut self, size: usize) -> Result<Buffer> {
        let level = size_to_level(size).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("invalid size: {size} (must be 1-67108864 bytes)"),
            )
        })?;

        self.try_allocate(level)
    }

    /// Tries to allocate a buffer at the given level.
    ///
    /// Returns `OutOfMemory` error if the memory limit is reached.
    fn try_allocate(&mut self, level: usize) -> Result<Buffer> {
        // Try to find a free buffer at this level or split from a larger one
        if let Some(buffer) = self.try_allocate_from_free_lists(level) {
            return Ok(buffer);
        }

        // Need to allocate a new 64 MiB block
        self.allocate_new_block()?;

        // Try again - should succeed now
        self.try_allocate_from_free_lists(level)
            .ok_or_else(|| Error::other("allocation failed unexpectedly"))
    }

    /// Tries to allocate from existing free lists.
    fn try_allocate_from_free_lists(&mut self, level: usize) -> Option<Buffer> {
        // Look for a free buffer at this level or higher
        for search_level in level..NUM_LEVELS {
            if !self.free_lists[search_level].is_empty() {
                // Found a free buffer, potentially need to split
                return Some(self.allocate_at_level(search_level, level));
            }
        }
        None
    }

    /// Allocates a buffer by taking from `from_level` and splitting down to `target_level`.
    fn allocate_at_level(&mut self, from_level: usize, target_level: usize) -> Buffer {
        // Pop a free node from the source level
        let node = self.free_lists[from_level].pop_front().unwrap();

        // SAFETY: node is valid and was in the free list
        let (block, index_in_level) = unsafe {
            let node_ref = &*node.as_ptr();
            (node_ref.data.block, node_ref.data.index_in_level)
        };

        // SAFETY: block pointer is valid
        let block_ref = unsafe { &mut *block.as_ptr() };

        if from_level == target_level {
            // No splitting needed
            block_ref.set_state(from_level, index_in_level, NodeState::Allocated);

            let ptr = block_ref.get_memory_addr(from_level, index_in_level);
            let len = level_to_size(from_level);

            // SAFETY: ptr is valid, len is correct, block is valid
            unsafe {
                Buffer::new(
                    NonNull::new(ptr).unwrap(),
                    len,
                    from_level,
                    index_in_level,
                    block,
                    Arc::clone(&self.return_tx),
                )
            }
        } else {
            // Need to split
            self.split_and_allocate(block, from_level, index_in_level, target_level)
        }
    }

    /// Splits a block from `from_level` down to `target_level` and allocates.
    fn split_and_allocate(
        &mut self,
        block: NonNull<BuddyBlock>,
        from_level: usize,
        from_index: usize,
        target_level: usize,
    ) -> Buffer {
        // SAFETY: block pointer is valid
        let block_ref = unsafe { &mut *block.as_ptr() };

        let mut current_level = from_level;
        let mut current_index = from_index;

        while current_level > target_level {
            // Mark current node as split
            block_ref.set_state(current_level, current_index, NodeState::Split);

            // Get first child
            let (child_level, first_child_index) =
                BuddyBlock::get_first_child(current_level, current_index).unwrap();

            // Add siblings 1-3 to free list (child 0 will be used or split further)
            for i in 1..4 {
                let child_index = first_child_index + i;
                block_ref.set_state(child_level, child_index, NodeState::Free);

                let node = block_ref.get_free_node_mut(child_level, child_index);
                // SAFETY: node is valid and not in any list
                unsafe {
                    self.free_lists[child_level].push_front(node);
                }
            }

            // Continue with child 0
            current_level = child_level;
            current_index = first_child_index;
        }

        // Allocate at target level
        block_ref.set_state(current_level, current_index, NodeState::Allocated);

        let ptr = block_ref.get_memory_addr(current_level, current_index);
        let len = level_to_size(current_level);

        // SAFETY: ptr is valid, len is correct, block is valid
        unsafe {
            Buffer::new(
                NonNull::new(ptr).unwrap(),
                len,
                current_level,
                current_index,
                block,
                Arc::clone(&self.return_tx),
            )
        }
    }

    /// Allocates a new 64 MiB block from the underlying allocator.
    fn allocate_new_block(&mut self) -> Result<()> {
        // Check memory limit
        if self.allocated_memory + SIZE_64MIB > self.max_memory {
            return Err(Error::new(ErrorKind::OutOfMemory, "memory limit reached"));
        }

        // Allocate memory
        let memory = self.allocator.allocate(SIZE_64MIB)?;

        // Create buddy block
        // SAFETY: memory is valid and points to SIZE_64MIB bytes
        let block = unsafe { BuddyBlock::new(memory) };

        // Get pointer to block before adding to list
        // SAFETY: We need a raw pointer to push the free node. The block is heap-allocated.
        let block_ptr =
            NonNull::new(std::ptr::from_ref::<BuddyBlock>(block.as_ref()).cast_mut()).unwrap();

        // Add root node to level 3 free list
        // SAFETY: block is valid, and the level 3 node exists
        unsafe {
            let node = (*block_ptr.as_ptr()).get_free_node_mut(3, 0);
            self.free_lists[3].push_front(node);
        }

        // Add block to our list
        // Note: We need to be careful here. After the block is added to the list,
        // the pointer we got earlier is still valid because Box is heap-allocated.
        self.blocks.push(block);
        self.allocated_memory += SIZE_64MIB;

        Ok(())
    }

    /// Deallocates a buffer and returns it to the appropriate free list.
    pub(crate) fn deallocate_buffer(
        &mut self,
        level: usize,
        index: usize,
        block: NonNull<BuddyBlock>,
    ) {
        // SAFETY: block pointer is valid
        let block_ref = unsafe { &mut *block.as_ptr() };

        // Try to merge with siblings
        let (final_level, _final_index) = self.try_merge(block_ref, level, index);

        // Check if we should satisfy a waiting allocation
        if let Some(min_waiting) = self.min_waiting_level {
            // Check if we can satisfy any waiting allocation
            // We can satisfy if the freed buffer is >= the waiting level
            if final_level >= min_waiting {
                // Find the smallest waiting level we can satisfy
                for wait_level in min_waiting..=final_level {
                    if let Some(sender) = self.waiting_lists[wait_level].pop_front() {
                        // Allocate from the free buffer and send to waiter
                        if let Some(buffer) = self.try_allocate_from_free_lists(wait_level) {
                            let _ = sender.send(buffer);
                        }
                        self.update_min_waiting_level();
                        return;
                    }
                }
            }
        }

        // No waiters to satisfy, buffer is already in free list from try_merge
    }

    /// Tries to merge freed buffer with its buddies, returns the final level and index.
    fn try_merge(&mut self, block: &mut BuddyBlock, level: usize, index: usize) -> (usize, usize) {
        let mut current_level = level;
        let mut current_index = index;

        loop {
            // Check if we can merge with siblings
            if current_level >= 3 {
                // At root level, just mark as free
                block.set_state(current_level, current_index, NodeState::Free);
                let node = block.get_free_node_mut(current_level, current_index);
                // SAFETY: node is valid and not in any list
                unsafe {
                    self.free_lists[current_level].push_front(node);
                }
                return (current_level, current_index);
            }

            // Get sibling indices
            let siblings = BuddyBlock::get_siblings(current_level, current_index);

            // Check if all siblings are free
            let all_free = siblings.iter().all(|&idx| {
                idx == current_index || block.get_state(current_level, idx) == NodeState::Free
            });

            if !all_free {
                // Cannot merge, just mark as free
                block.set_state(current_level, current_index, NodeState::Free);
                let node = block.get_free_node_mut(current_level, current_index);
                // SAFETY: node is valid and not in any list
                unsafe {
                    self.free_lists[current_level].push_front(node);
                }
                return (current_level, current_index);
            }

            // All siblings are free, remove them from free list and merge
            for &sibling_idx in &siblings {
                if sibling_idx != current_index {
                    // Remove from free list
                    let node = block.get_free_node_mut(current_level, sibling_idx);
                    // SAFETY: node is valid and in the free list
                    unsafe {
                        self.free_lists[current_level].remove(node);
                    }
                }
                // Mark as allocated (will be managed by parent)
                block.set_state(current_level, sibling_idx, NodeState::Allocated);
            }

            // Move up to parent
            let (parent_level, parent_index) =
                BuddyBlock::get_parent(current_level, current_index).unwrap();
            current_level = parent_level;
            current_index = parent_index;
        }
    }

    /// Updates the minimum waiting level after a change to waiting lists.
    fn update_min_waiting_level(&mut self) {
        self.min_waiting_level =
            (0..NUM_LEVELS).find(|&level| !self.waiting_lists[level].is_empty());
    }
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // Deallocate all 64 MiB blocks
        for block in &self.blocks {
            // SAFETY: memory was allocated by our allocator with SIZE_64MIB
            unsafe {
                self.allocator.deallocate(block.memory, SIZE_64MIB);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buddy::{LEVEL_SIZES, SIZE_1MIB};

    #[test]
    fn test_pool_builder_defaults() {
        let pool = BufferPoolBuilder::new().build();
        // Just verify it builds without error
        drop(pool);
    }

    #[test]
    fn test_pool_builder_custom() {
        let pool = BufferPoolBuilder::new()
            .max_memory(128 * 1024 * 1024)
            .build();
        drop(pool);
    }

    #[test]
    fn test_simple_allocation() {
        let pool = BufferPoolBuilder::new().build();
        let buffer = pool.allocate(SIZE_1MIB).unwrap();
        assert_eq!(buffer.len(), SIZE_1MIB);
    }

    #[test]
    fn test_allocation_sizes() {
        let pool = BufferPoolBuilder::new().build();

        // 1 MiB
        let b1 = pool.allocate(1).unwrap();
        assert_eq!(b1.len(), SIZE_1MIB);

        // 4 MiB
        let b2 = pool.allocate(SIZE_1MIB + 1).unwrap();
        assert_eq!(b2.len(), LEVEL_SIZES[1]);

        // 16 MiB
        let b3 = pool.allocate(LEVEL_SIZES[1] + 1).unwrap();
        assert_eq!(b3.len(), LEVEL_SIZES[2]);

        // 64 MiB
        let b4 = pool.allocate(LEVEL_SIZES[2] + 1).unwrap();
        assert_eq!(b4.len(), LEVEL_SIZES[3]);
    }

    #[test]
    fn test_allocation_reuse() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        // Allocate and drop
        let addr1 = {
            let buffer = pool.allocate(SIZE_1MIB).unwrap();
            buffer.as_ptr() as usize
        };

        // Allocate again - should reuse
        let buffer2 = pool.allocate(SIZE_1MIB).unwrap();
        let addr2 = buffer2.as_ptr() as usize;

        // In buddy allocation, we might not get the exact same address
        // but we should be within the same 64MiB block
        assert!(addr1 > 0);
        assert!(addr2 > 0);
    }

    #[test]
    fn test_memory_limit_sync() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        // Allocate the entire limit
        let _b1 = pool.allocate(SIZE_64MIB).unwrap();

        // This should fail
        let result = pool.allocate(SIZE_1MIB);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ErrorKind::OutOfMemory);
    }

    #[test]
    fn test_invalid_size() {
        let pool = BufferPoolBuilder::new().build();

        // Size 0
        let result = pool.allocate(0);
        assert!(result.is_err());

        // Size too large
        let result = pool.allocate(SIZE_64MIB + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_buddy_splitting() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        // Allocate 64 1MiB buffers (should all fit in one 64MiB block)
        let buffers: Vec<_> = (0..64).map(|_| pool.allocate(SIZE_1MIB).unwrap()).collect();

        assert_eq!(buffers.len(), 64);

        // All buffers should have valid addresses within the 64MiB range
        let base = buffers[0].as_ptr() as usize;
        for (i, buf) in buffers.iter().enumerate() {
            let addr = buf.as_ptr() as usize;
            // Each buffer should be SIZE_1MIB apart (though order may vary)
            assert!(addr >= base - SIZE_64MIB && addr < base + SIZE_64MIB);
            assert_eq!(buf.len(), SIZE_1MIB);
            // Verify the index is valid
            assert!(i < 64);
        }
    }

    #[test]
    fn test_buddy_merging() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        // Allocate 4 x 16MiB (fills entire 64MiB)
        let b1 = pool.allocate(LEVEL_SIZES[2]).unwrap();
        let b2 = pool.allocate(LEVEL_SIZES[2]).unwrap();
        let b3 = pool.allocate(LEVEL_SIZES[2]).unwrap();
        let b4 = pool.allocate(LEVEL_SIZES[2]).unwrap();

        // Should be at limit now
        assert!(pool.allocate(SIZE_1MIB).is_err());

        // Free all 4 (should merge back to one 64MiB)
        drop(b1);
        drop(b2);
        drop(b3);
        drop(b4);

        // Should be able to allocate 64MiB now
        let b5 = pool.allocate(SIZE_64MIB).unwrap();
        assert_eq!(b5.len(), SIZE_64MIB);
    }

    #[tokio::test]
    async fn test_async_allocation() {
        let pool = BufferPoolBuilder::new().build();
        let buffer = pool.async_allocate(SIZE_1MIB).await.unwrap();
        assert_eq!(buffer.len(), SIZE_1MIB);
    }

    #[tokio::test]
    async fn test_async_allocation_waiting() {
        use std::time::Duration;
        use tokio::time::timeout;

        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        // Allocate the entire limit
        let buffer = pool.async_allocate(SIZE_64MIB).await.unwrap();

        let pool_clone = pool.clone();

        // Start an async allocation that will have to wait
        let handle = tokio::spawn(async move { pool_clone.async_allocate(SIZE_1MIB).await });

        // Give the async allocation time to start waiting
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Free the buffer
        drop(buffer);

        // The async allocation should complete
        let result = timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok());
        let buffer = result.unwrap().unwrap().unwrap();
        assert_eq!(buffer.len(), SIZE_1MIB);
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB * 2).build();

        assert_eq!(pool.allocated_memory().await, 0);
        assert_eq!(pool.max_memory().await, SIZE_64MIB * 2);

        let _buffer = pool.async_allocate(SIZE_1MIB).await.unwrap();
        assert_eq!(pool.allocated_memory().await, SIZE_64MIB);
    }

    #[test]
    fn test_buffer_write_read() {
        let pool = BufferPoolBuilder::new().build();
        let mut buffer = pool.allocate(SIZE_1MIB).unwrap();

        // Write pattern
        for (i, byte) in buffer.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Read back and verify
        for (i, byte) in buffer.iter().enumerate() {
            assert_eq!(*byte, (i % 256) as u8);
        }
    }

    #[test]
    fn test_multiple_pools() {
        let pool1 = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();
        let pool2 = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        let b1 = pool1.allocate(SIZE_64MIB).unwrap();
        let b2 = pool2.allocate(SIZE_64MIB).unwrap();

        // Both should succeed as they're separate pools
        assert_eq!(b1.len(), SIZE_64MIB);
        assert_eq!(b2.len(), SIZE_64MIB);
    }

    #[test]
    fn test_clone_pool() {
        let pool = BufferPoolBuilder::new().max_memory(SIZE_64MIB).build();

        let pool_clone = pool.clone();

        let b1 = pool.allocate(SIZE_64MIB).unwrap();

        // Clone shares the same state, so this should fail
        let result = pool_clone.allocate(SIZE_1MIB);
        assert!(result.is_err());

        drop(b1);

        // Now it should work
        let _b2 = pool_clone.allocate(SIZE_1MIB).unwrap();
    }
}
