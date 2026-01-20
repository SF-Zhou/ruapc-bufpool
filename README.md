# ruapc-bufpool

[![Crates.io](https://img.shields.io/crates/v/ruapc-bufpool.svg)](https://crates.io/crates/ruapc-bufpool)
[![Documentation](https://docs.rs/ruapc-bufpool/badge.svg)](https://docs.rs/ruapc-bufpool)
[![License](https://img.shields.io/crates/l/ruapc-bufpool.svg)](https://github.com/SF-Zhou/ruapc-bufpool#license)
[![Build Status](https://github.com/SF-Zhou/ruapc-bufpool/workflows/CI/badge.svg)](https://github.com/SF-Zhou/ruapc-bufpool/actions)

A high-performance memory pool using the buddy memory allocation algorithm for efficient fixed-size buffer management. This crate is part of the [ruapc](https://github.com/SF-Zhou/ruapc) project.

## Features

- **Buddy Memory Allocation**: Efficiently manages memory in 64 MiB blocks, supporting allocation of 1 MiB, 4 MiB, 16 MiB, and 64 MiB buffers
- **Both Sync and Async APIs**: Designed for tokio environments with async-first design using `tokio::sync::Mutex`
- **Automatic Memory Reclamation**: Buffers are automatically returned to the pool when dropped
- **Memory Limits**: Configurable maximum memory usage with async waiting when limits are reached
- **Custom Allocators**: Pluggable allocator trait for custom memory allocation backends
- **O(1) Buddy Merging**: Intrusive doubly-linked list with O(1) free/merge operations
- **Zero Copy**: Direct memory access through `Buffer` type that implements `Deref<Target = [u8]>`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ruapc-bufpool = "0.1"
```

## Quick Start

### Synchronous Allocation

```rust
use ruapc_bufpool::{BufferPool, BufferPoolBuilder};

fn main() -> std::io::Result<()> {
    // Create a buffer pool with 256 MiB max memory (default)
    let pool = BufferPoolBuilder::new()
        .max_memory(256 * 1024 * 1024)
        .build();

    // Allocate a 1 MiB buffer
    let mut buffer = pool.allocate(1024 * 1024)?;
    
    // Write to the buffer
    buffer[0] = 42;
    buffer[1] = 43;
    
    // Read from the buffer
    assert_eq!(buffer[0], 42);
    
    // Buffer is automatically returned to the pool when dropped
    drop(buffer);
    
    Ok(())
}
```

### Asynchronous Allocation

```rust
use ruapc_bufpool::BufferPoolBuilder;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let pool = BufferPoolBuilder::new()
        .max_memory(128 * 1024 * 1024)
        .build();

    // Allocate asynchronously - will wait if memory limit is reached
    let buffer = pool.async_allocate(4 * 1024 * 1024).await?;
    
    assert!(buffer.len() >= 4 * 1024 * 1024);
    
    Ok(())
}
```

### Custom Allocator

```rust
use ruapc_bufpool::{Allocator, BufferPoolBuilder};
use std::io::Result;

struct MyAllocator;

impl Allocator for MyAllocator {
    fn allocate(&self, size: usize) -> Result<*mut u8> {
        // Custom allocation logic
        // ...
        # unimplemented!()
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        // Custom deallocation logic
        // ...
    }
}

fn main() {
    let pool = BufferPoolBuilder::new()
        .allocator(Box::new(MyAllocator))
        .build();
}
```

## Architecture

### Buddy Memory Allocation

The pool uses a buddy memory allocation algorithm with 4 levels:

| Level | Size | Blocks per 64 MiB |
|-------|------|-------------------|
| 0 | 1 MiB | 64 |
| 1 | 4 MiB | 16 |
| 2 | 16 MiB | 4 |
| 3 | 64 MiB | 1 |

When allocating:
1. Find the smallest level that can satisfy the request
2. If no free buffer at that level, look for a larger one
3. Split larger buffers into 4 smaller ones as needed
4. Return the first available buffer

When freeing:
1. Mark the buffer as free
2. Check if all 4 sibling buffers are free
3. If so, merge them into a larger buffer
4. Repeat until no more merging is possible

### Memory Management

- Each 64 MiB block maintains a state array tracking allocation status
- Free lists use intrusive doubly-linked lists for O(1) operations
- Buffer return uses unbounded channels to avoid blocking in drop

### Thread Safety

- Uses `tokio::sync::Mutex` for synchronization
- Sync allocation uses `blocking_lock()`
- Async allocation uses `lock().await`
- Buffer drop sends through unbounded channel (non-blocking)

## Performance Considerations

- **Allocation**: O(log n) where n is the number of levels (constant in practice)
- **Deallocation**: O(log n) for merging, O(1) for just freeing
- **Memory overhead**: ~85 bytes state array per 64 MiB block
- **Best for**: Large buffers (1+ MiB), frequent allocation/deallocation patterns

## API Reference

See the [documentation on docs.rs](https://docs.rs/ruapc-bufpool) for detailed API reference.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
