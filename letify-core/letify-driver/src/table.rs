//! Virtual handles.
//!
//! An allocation has to return a pointer immediately. Waiting for the agent to answer
//! would put a round trip in front of every `cuMemAlloc`, and PyTorch's caching
//! allocator calls that often enough for it to matter.
//!
//! So the local driver hands out its own pointers from a range that cannot be confused with a
//! real address, records what they stand for, and lets the agent reconcile them in the
//! background. A later call that names one carries the handle, not the address.
//!
//! The one thing this costs is honest failure. The caching allocator normally learns that
//! the device is full when `cuMemAlloc` returns an error, frees its cache and retries.
//! With a virtual pointer there is nothing to fail yet, so the local driver keeps its own
//! accounting of device memory and refuses locally once the budget is gone. That keeps
//! the retry path working instead of turning an out of memory condition into a crash at
//! the next synchronization.

use std::collections::HashMap;
use std::sync::Mutex;

/// Virtual pointers start here. Real device addresses on every supported platform are
/// far below this, so a value in this range is unambiguously one of ours.
const BASE: u64 = 0x7000_0000_0000_0000;

/// What one virtual pointer stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub handle: u64,
    pub bytes: u64,
}

/// Every handle the local driver has issued, plus its memory accounting.
#[derive(Debug)]
pub struct Table {
    next: u64,
    allocations: HashMap<u64, Allocation>,
    /// Device memory the agent reported, and how much of it is committed.
    total_bytes: u64,
    used_bytes: u64,
    /// Held back for the driver's own context, library workspaces and fragmentation.
    reserve_bytes: u64,
}

impl Table {
    pub fn new() -> Self {
        Table {
            next: 1,
            allocations: HashMap::new(),
            total_bytes: 0,
            used_bytes: 0,
            reserve_bytes: 512 * 1024 * 1024,
        }
    }

    /// Record the device size, once the agent has said what it is.
    pub fn set_total(&mut self, total_bytes: u64) {
        self.total_bytes = total_bytes;
    }

    pub fn free_bytes(&self) -> u64 {
        self.total_bytes
            .saturating_sub(self.used_bytes)
            .saturating_sub(self.reserve_bytes)
    }

    /// Reserve memory and return the virtual pointer for it.
    ///
    /// Returns `None` when the budget is gone, which is what lets the caller's own
    /// retry path run instead of failing later at a synchronization.
    pub fn allocate(&mut self, bytes: u64) -> Option<(u64, u64)> {
        if self.total_bytes > 0 && bytes > self.free_bytes() {
            return None;
        }
        let handle = self.next;
        self.next += 1;
        self.used_bytes += bytes;
        self.allocations.insert(handle, Allocation { handle, bytes });
        Some((pointer_of(handle), handle))
    }

    /// Release a virtual pointer, returning the handle the agent knows it by.
    pub fn free(&mut self, pointer: u64) -> Option<u64> {
        let handle = handle_of(pointer)?;
        let allocation = self.allocations.remove(&handle)?;
        self.used_bytes = self.used_bytes.saturating_sub(allocation.bytes);
        Some(handle)
    }

    /// Which allocation a pointer falls inside, and how far into it.
    ///
    /// Callers do pointer arithmetic on device memory, so a copy often names an address
    /// partway through an allocation rather than its start.
    pub fn resolve(&self, pointer: u64) -> Option<(u64, u64)> {
        let handle = handle_of(pointer)?;
        self.allocations.get(&handle)?;
        Some((handle, offset_of(pointer)))
    }

    /// What one handle stands for, for the report the Python side reads.
    pub fn allocation(&self, handle: u64) -> Option<Allocation> {
        self.allocations.get(&handle).copied()
    }

    pub fn live(&self) -> usize {
        self.allocations.len()
    }

    pub fn used(&self) -> u64 {
        self.used_bytes
    }
}

impl Default for Table {
    fn default() -> Self {
        Table::new()
    }
}

/// The virtual pointer that stands for a handle.
///
/// Handles are spaced a gigabyte apart so that pointer arithmetic inside one allocation
/// stays inside its own range and can be attributed back to it.
pub fn pointer_of(handle: u64) -> u64 {
    BASE + handle * 0x4000_0000
}

/// The handle a virtual pointer belongs to, or `None` if it is not one of ours.
pub fn handle_of(pointer: u64) -> Option<u64> {
    if pointer < BASE {
        return None;
    }
    Some((pointer - BASE) / 0x4000_0000)
}

/// The offset into its allocation that a pointer names.
pub fn offset_of(pointer: u64) -> u64 {
    if pointer < BASE {
        return 0;
    }
    (pointer - BASE) % 0x4000_0000
}

/// The process wide table. One device, one table, so a mutex is enough.
pub fn table() -> &'static Mutex<Table> {
    static TABLE: std::sync::OnceLock<Mutex<Table>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(Table::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pointer_maps_back_to_its_handle() {
        let pointer = pointer_of(3);
        assert_eq!(handle_of(pointer), Some(3));
        assert_eq!(offset_of(pointer), 0);
        assert_eq!(offset_of(pointer + 128), 128);
        assert_eq!(handle_of(pointer + 128), Some(3));
    }

    #[test]
    fn a_real_address_is_not_mistaken_for_ours() {
        assert_eq!(handle_of(0x7f00_1234), None);
        assert_eq!(handle_of(0), None);
    }

    #[test]
    fn allocation_accounting_survives_free() {
        let mut table = Table::new();
        table.set_total(8 * 1024 * 1024 * 1024);
        let (pointer, handle) = table.allocate(1024).unwrap();
        assert_eq!(table.live(), 1);
        assert_eq!(table.used(), 1024);
        assert_eq!(table.free(pointer), Some(handle));
        assert_eq!(table.live(), 0);
        assert_eq!(table.used(), 0);
    }

    #[test]
    fn the_budget_refuses_locally_so_the_caller_can_retry() {
        // This is the point of tracking memory here: the caching allocator learns the
        // device is full at the call, the way it would with a real driver, instead of
        // finding out at the next synchronization when it can no longer react.
        let mut table = Table::new();
        table.set_total(1024 * 1024 * 1024);
        assert!(table.allocate(2 * 1024 * 1024 * 1024).is_none());
        assert!(table.allocate(16 * 1024).is_some());
    }

    #[test]
    fn a_pointer_partway_in_resolves_to_its_offset() {
        let mut table = Table::new();
        let (pointer, handle) = table.allocate(4096).unwrap();
        assert_eq!(table.resolve(pointer + 512), Some((handle, 512)));
        assert_eq!(table.allocation(handle).unwrap().bytes, 4096);
        assert_eq!(table.resolve(0x1000), None);
    }

    #[test]
    fn with_no_reported_size_nothing_is_refused() {
        // Before the agent has answered, refusing would be guessing.
        let mut table = Table::new();
        assert!(table.allocate(64 * 1024 * 1024 * 1024).is_some());
    }
}
