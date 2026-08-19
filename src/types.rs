/// Node index into the model's node space. Indexes, not pointers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub u32);

/// Vehicle index into the model's fleet.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VehicleId(pub u32);

/// Scaled integer cost. No floats: ties and comparisons must be deterministic.
pub type Cost = i64;

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl VehicleId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
