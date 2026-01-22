//! Buddy memory allocation implementation.
//!
//! This module implements the buddy memory allocation algorithm with 4 levels:
//! - Level 0: 1MiB (1 block per parent = 64 blocks per 64MiB)
//! - Level 1: 4MiB (4 blocks per parent = 16 blocks per 64MiB)
//! - Level 2: 16MiB (4 blocks per parent = 4 blocks per 64MiB)
//! - Level 3: 64MiB (the root level = 1 block per 64MiB)
//!
//! Each 64MiB block has a state array tracking the allocation status of all nodes
//! in the buddy tree.

use std::ptr::NonNull;

use crate::intrusive_list::IntrusiveNode;

/// Size constants for each level.
pub const SIZE_1MIB: usize = 1024 * 1024;
pub const SIZE_4MIB: usize = 4 * SIZE_1MIB;
pub const SIZE_16MIB: usize = 4 * SIZE_4MIB;
pub const SIZE_64MIB: usize = 4 * SIZE_16MIB;

/// Number of levels in the buddy allocator.
pub const NUM_LEVELS: usize = 4;

/// Sizes for each level (indexed by level).
pub const LEVEL_SIZES: [usize; NUM_LEVELS] = [SIZE_1MIB, SIZE_4MIB, SIZE_16MIB, SIZE_64MIB];

/// Number of nodes at each level within a 64MiB block.
/// Level 0: 64 nodes (1MiB each)
/// Level 1: 16 nodes (4MiB each)
/// Level 2: 4 nodes (16MiB each)
/// Level 3: 1 node (64MiB)
#[allow(dead_code)]
pub const NODES_PER_LEVEL: [usize; NUM_LEVELS] = [64, 16, 4, 1];

/// Total number of nodes in the state array: 64 + 16 + 4 + 1 = 85
pub const TOTAL_STATE_NODES: usize = 85;

#[allow(clippy::manual_div_ceil)]
pub const STATE_ARRAY_BYTES: usize = (TOTAL_STATE_NODES * 2).div_ceil(8);

/// Starting index in the state array for each level.
/// Level 0: 0..64
/// Level 1: 64..80
/// Level 2: 80..84
/// Level 3: 84..85
pub const LEVEL_STATE_OFFSETS: [usize; NUM_LEVELS] = [0, 64, 80, 84];

/// State of a node in the buddy tree.
///
/// States are bit-packed into the buddy block's state array.
/// Each node uses 2 bits (00=Allocated, 01=Free, 10=Split).
/// This compact representation reduces memory overhead from 85 bytes to 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[derive(Default)]
pub enum NodeState {
    /// The node is allocated and in use.
    #[default]
    Allocated = 0,
    /// The node is free and available for allocation.
    Free = 1,
    /// The node has been split into smaller children.
    Split = 2,
}

impl NodeState {
    /// Converts state to its bit-packed representation.
    #[inline]
    pub const fn as_bits(self) -> u8 {
        self as u8
    }

    /// Creates state from its bit-packed representation.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Allocated,
            1 => Self::Free,
            2 => Self::Split,
            _ => panic!("invalid bits"),
        }
    }
}

/// Data stored in each free list node.
///
/// This contains the information needed to identify which block and which
/// position within the block this free node represents.
#[derive(Debug)]
pub struct FreeNodeData {
    /// Pointer to the parent `BuddyBlock`.
    pub block: NonNull<BuddyBlock>,
    /// Index within the level (0-based).
    pub index_in_level: usize,
}

// SAFETY: FreeNodeData only contains NonNull which is Send if the pointed type is Send
unsafe impl Send for FreeNodeData {}

// SAFETY: FreeNodeData only contains NonNull which is Sync if the pointed type is Sync
unsafe impl Sync for FreeNodeData {}

/// A free list node that can be inserted into an intrusive list.
pub type FreeNode = IntrusiveNode<FreeNodeData>;

/// A 64MiB buddy block that manages memory allocation at all levels.
///
/// Each block contains:
/// - A pointer to the raw 64MiB memory region
/// - A bit-packed state array tracking allocation status of all nodes
/// - Free list nodes for each possible allocation unit
///
/// The free list nodes are stored inline to avoid additional allocations.
pub struct BuddyBlock {
    /// Pointer to the 64MiB memory region.
    pub memory: *mut u8,

    /// Bit-packed allocation state for all nodes in the buddy tree.
    pub states: [u8; STATE_ARRAY_BYTES],

    /// Free list nodes for level 0 (64 nodes of 1MiB each).
    pub level0_nodes: [FreeNode; 64],

    /// Free list nodes for level 1 (16 nodes of 4MiB each).
    pub level1_nodes: [FreeNode; 16],

    /// Free list nodes for level 2 (4 nodes of 16MiB each).
    pub level2_nodes: [FreeNode; 4],

    /// Free list node for level 3 (1 node of 64MiB).
    pub level3_node: FreeNode,
}

impl BuddyBlock {
    /// Creates a new `BuddyBlock` managing the given memory region.
    ///
    /// The block is initialized with the entire 64MiB region marked as free.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `memory` points to a valid 64MiB region
    /// - The memory region will remain valid for the lifetime of this block
    pub unsafe fn new(memory: *mut u8) -> Box<Self> {
        const fn placeholder_data() -> FreeNodeData {
            FreeNodeData {
                block: NonNull::dangling(),
                index_in_level: 0,
            }
        }

        let mut block = Box::new(Self {
            memory,
            states: [0u8; STATE_ARRAY_BYTES],
            level0_nodes: std::array::from_fn(|_| FreeNode::new(placeholder_data())),
            level1_nodes: std::array::from_fn(|_| FreeNode::new(placeholder_data())),
            level2_nodes: std::array::from_fn(|_| FreeNode::new(placeholder_data())),
            level3_node: FreeNode::new(placeholder_data()),
        });

        // Get a non-null pointer to the block
        let block_ptr = NonNull::new(std::ptr::from_mut::<Self>(block.as_mut())).unwrap();

        // Initialize all free list nodes with correct back-pointers
        for (i, node) in block.level0_nodes.iter_mut().enumerate() {
            node.data = FreeNodeData {
                block: block_ptr,
                index_in_level: i,
            };
        }
        for (i, node) in block.level1_nodes.iter_mut().enumerate() {
            node.data = FreeNodeData {
                block: block_ptr,
                index_in_level: i,
            };
        }
        for (i, node) in block.level2_nodes.iter_mut().enumerate() {
            node.data = FreeNodeData {
                block: block_ptr,
                index_in_level: i,
            };
        }
        block.level3_node.data = FreeNodeData {
            block: block_ptr,
            index_in_level: 0,
        };

        // Initialize all states to Allocated (0x00)
        for byte in &mut block.states {
            *byte = 0;
        }

        // Set root node to Free using bit-packing
        block.set_state(3, 0, NodeState::Free);

        block
    }

    /// Gets a mutable pointer to the free node for the given level and index.
    ///
    /// # Panics
    ///
    /// Panics if level or index is out of bounds.
    pub fn get_free_node_mut(&mut self, level: usize, index: usize) -> NonNull<FreeNode> {
        let node = match level {
            0 => &mut self.level0_nodes[index],
            1 => &mut self.level1_nodes[index],
            2 => &mut self.level2_nodes[index],
            3 => {
                debug_assert_eq!(index, 0);
                &mut self.level3_node
            }
            _ => panic!("invalid level: {level}"),
        };
        NonNull::new(node).unwrap()
    }

    /// Gets the state array index for a node at the given level and index.
    #[inline]
    pub const fn state_index(level: usize, index: usize) -> usize {
        LEVEL_STATE_OFFSETS[level] + index
    }

    /// Gets the state of a node at the given level and index.
    #[inline]
    pub const fn get_state(&self, level: usize, index: usize) -> NodeState {
        let idx = Self::state_index(level, index);
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = self.states[byte_idx];
        let mask = 0b11 << bit_offset;
        NodeState::from_bits((bits & mask) >> bit_offset)
    }

    /// Sets the state of a node at the given level and index.
    #[inline]
    pub const fn set_state(&mut self, level: usize, index: usize, state: NodeState) {
        let idx = Self::state_index(level, index);
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = self.states[byte_idx];
        let mask = !(0b11 << bit_offset);
        let new_bits = (bits & mask) | (state.as_bits() << bit_offset);
        self.states[byte_idx] = new_bits;
    }

    /// Gets the memory address for a node at the given level and index.
    #[inline]
    pub const fn get_memory_addr(&self, level: usize, index: usize) -> *mut u8 {
        let offset = index * LEVEL_SIZES[level];
        // SAFETY: offset is within the 64MiB block
        unsafe { self.memory.add(offset) }
    }

    /// Gets the parent level and index for a node.
    ///
    /// Returns `None` for level 3 (root) nodes.
    #[inline]
    pub const fn get_parent(level: usize, index: usize) -> Option<(usize, usize)> {
        if level >= 3 {
            None
        } else {
            Some((level + 1, index / 4))
        }
    }

    /// Gets the sibling indices for a node.
    ///
    /// Returns the indices of all 4 siblings (including the node itself).
    #[inline]
    pub const fn get_siblings(index: usize) -> [usize; 4] {
        let base = (index / 4) * 4;
        [base, base + 1, base + 2, base + 3]
    }

    /// Gets the first child index for a node.
    ///
    /// Returns `None` for level 0 (leaf) nodes.
    #[inline]
    pub const fn get_first_child(level: usize, index: usize) -> Option<(usize, usize)> {
        if level == 0 {
            None
        } else {
            Some((level - 1, index * 4))
        }
    }
}

// SAFETY: BuddyBlock can be sent between threads. The FreeNode fields contain
// raw pointers that are only accessed while holding the pool's mutex lock.
// The block and all its nodes are protected by external synchronization.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for BuddyBlock {}

// SAFETY: BuddyBlock can be shared between threads (with external synchronization)
unsafe impl Sync for BuddyBlock {}

/// Calculates the allocation level for a given size.
///
/// Returns `None` if the size exceeds the maximum (64MiB).
#[inline]
pub const fn size_to_level(size: usize) -> Option<usize> {
    if size == 0 {
        return None;
    }
    if size <= SIZE_1MIB {
        Some(0)
    } else if size <= SIZE_4MIB {
        Some(1)
    } else if size <= SIZE_16MIB {
        Some(2)
    } else if size <= SIZE_64MIB {
        Some(3)
    } else {
        None
    }
}

/// Gets the actual allocation size for a given level.
#[inline]
#[allow(dead_code)] // Used in tests
pub const fn level_to_size(level: usize) -> usize {
    LEVEL_SIZES[level]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_to_level() {
        assert_eq!(size_to_level(0), None);
        assert_eq!(size_to_level(1), Some(0));
        assert_eq!(size_to_level(SIZE_1MIB), Some(0));
        assert_eq!(size_to_level(SIZE_1MIB + 1), Some(1));
        assert_eq!(size_to_level(SIZE_4MIB), Some(1));
        assert_eq!(size_to_level(SIZE_4MIB + 1), Some(2));
        assert_eq!(size_to_level(SIZE_16MIB), Some(2));
        assert_eq!(size_to_level(SIZE_16MIB + 1), Some(3));
        assert_eq!(size_to_level(SIZE_64MIB), Some(3));
        assert_eq!(size_to_level(SIZE_64MIB + 1), None);
    }

    #[test]
    fn test_level_to_size() {
        assert_eq!(level_to_size(0), SIZE_1MIB);
        assert_eq!(level_to_size(1), SIZE_4MIB);
        assert_eq!(level_to_size(2), SIZE_16MIB);
        assert_eq!(level_to_size(3), SIZE_64MIB);
    }

    #[test]
    fn test_state_index() {
        // Level 0: indices 0..64
        assert_eq!(BuddyBlock::state_index(0, 0), 0);
        assert_eq!(BuddyBlock::state_index(0, 63), 63);

        // Level 1: indices 64..80
        assert_eq!(BuddyBlock::state_index(1, 0), 64);
        assert_eq!(BuddyBlock::state_index(1, 15), 79);

        // Level 2: indices 80..84
        assert_eq!(BuddyBlock::state_index(2, 0), 80);
        assert_eq!(BuddyBlock::state_index(2, 3), 83);

        // Level 3: index 84
        assert_eq!(BuddyBlock::state_index(3, 0), 84);
    }

    #[test]
    fn test_get_parent() {
        // Level 0 nodes
        assert_eq!(BuddyBlock::get_parent(0, 0), Some((1, 0)));
        assert_eq!(BuddyBlock::get_parent(0, 3), Some((1, 0)));
        assert_eq!(BuddyBlock::get_parent(0, 4), Some((1, 1)));
        assert_eq!(BuddyBlock::get_parent(0, 63), Some((1, 15)));

        // Level 1 nodes
        assert_eq!(BuddyBlock::get_parent(1, 0), Some((2, 0)));
        assert_eq!(BuddyBlock::get_parent(1, 15), Some((2, 3)));

        // Level 2 nodes
        assert_eq!(BuddyBlock::get_parent(2, 0), Some((3, 0)));
        assert_eq!(BuddyBlock::get_parent(2, 3), Some((3, 0)));

        // Level 3 (root)
        assert_eq!(BuddyBlock::get_parent(3, 0), None);
    }

    #[test]
    fn test_get_siblings() {
        assert_eq!(BuddyBlock::get_siblings(0), [0, 1, 2, 3]);
        assert_eq!(BuddyBlock::get_siblings(2), [0, 1, 2, 3]);
        assert_eq!(BuddyBlock::get_siblings(4), [4, 5, 6, 7]);
        assert_eq!(BuddyBlock::get_siblings(0), [0, 1, 2, 3]);
        assert_eq!(BuddyBlock::get_siblings(5), [4, 5, 6, 7]);
    }

    #[test]
    fn test_get_first_child() {
        assert_eq!(BuddyBlock::get_first_child(0, 0), None);
        assert_eq!(BuddyBlock::get_first_child(1, 0), Some((0, 0)));
        assert_eq!(BuddyBlock::get_first_child(1, 1), Some((0, 4)));
        assert_eq!(BuddyBlock::get_first_child(2, 0), Some((1, 0)));
        assert_eq!(BuddyBlock::get_first_child(3, 0), Some((2, 0)));
    }

    #[test]
    fn test_buddy_block_creation() {
        // Allocate a 64MiB region for testing
        let layout = std::alloc::Layout::from_size_align(SIZE_64MIB, 4096).unwrap();
        let memory = unsafe { std::alloc::alloc(layout) };
        assert!(!memory.is_null());

        let block = unsafe { BuddyBlock::new(memory) };

        // Check that root is free
        assert_eq!(block.get_state(3, 0), NodeState::Free);

        // Check that all other nodes are allocated (default)
        for i in 0..64 {
            assert_eq!(block.get_state(0, i), NodeState::Allocated);
        }
        for i in 0..16 {
            assert_eq!(block.get_state(1, i), NodeState::Allocated);
        }
        for i in 0..4 {
            assert_eq!(block.get_state(2, i), NodeState::Allocated);
        }

        // Clean up
        unsafe {
            std::alloc::dealloc(memory, layout);
        }
    }

    #[test]
    fn test_memory_address_calculation() {
        let layout = std::alloc::Layout::from_size_align(SIZE_64MIB, 4096).unwrap();
        let memory = unsafe { std::alloc::alloc(layout) };
        assert!(!memory.is_null());

        let block = unsafe { BuddyBlock::new(memory) };

        // Level 3 (64MiB)
        assert_eq!(block.get_memory_addr(3, 0), memory);

        // Level 2 (16MiB each)
        assert_eq!(block.get_memory_addr(2, 0), memory);
        assert_eq!(block.get_memory_addr(2, 1), unsafe {
            memory.add(SIZE_16MIB)
        });

        // Level 1 (4MiB each)
        assert_eq!(block.get_memory_addr(1, 0), memory);
        assert_eq!(block.get_memory_addr(1, 4), unsafe {
            memory.add(SIZE_16MIB)
        });

        // Level 0 (1MiB each)
        assert_eq!(block.get_memory_addr(0, 0), memory);
        assert_eq!(block.get_memory_addr(0, 1), unsafe {
            memory.add(SIZE_1MIB)
        });

        unsafe {
            std::alloc::dealloc(memory, layout);
        }
    }
}
