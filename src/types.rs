#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub u32);

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
