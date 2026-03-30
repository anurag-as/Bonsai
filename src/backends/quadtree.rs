//! Loose 2^D-tree (Quadtree / Octree / Hexadecatree) spatial index backend.
//!
//! Generalises the 2D quadtree, 3D octree, and 4D hexadecatree under a single
//! generic struct parameterised by `D`. Each internal node has exactly `2^D`
//! children. The "loose" variant expands each node's effective bounds by 2×
//! the cell half-size so that a point near a boundary is always contained in
//! exactly one child's loose bounds.

use std::collections::HashMap;

use crate::backends::SpatialBackend;
use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

const LEAF_CAPACITY: usize = 8;
const MAX_DEPTH: usize = 20;

struct QuadNode<C, const D: usize> {
    bounds: BBox<C, D>,
    loose_bounds: BBox<C, D>,
    entries: Vec<(Point<C, D>, EntryId)>,
    children: Option<Vec<QuadNode<C, D>>>,
    depth: usize,
}

impl<C: CoordType, const D: usize> QuadNode<C, D> {
    fn new_leaf(bounds: BBox<C, D>, loose_bounds: BBox<C, D>, depth: usize) -> Self {
        Self {
            bounds,
            loose_bounds,
            entries: Vec::new(),
            children: None,
            depth,
        }
    }

    /// Assigns a point to one of the `2^D` children by comparing each axis
    /// coordinate against the node midpoint. Bit `d` is set if the point is
    /// on the upper half of axis `d`.
    fn child_index(&self, point: &Point<C, D>) -> usize {
        let mut idx = 0usize;
        for d in 0..D {
            let lo: f64 = self.bounds.min.coords()[d].into();
            let hi: f64 = self.bounds.max.coords()[d].into();
            let mid: C = C::from(((lo + hi) * 0.5) as f32);
            if point.coords()[d] >= mid {
                idx |= 1 << d;
            }
        }
        idx
    }

    fn make_children(&self) -> Vec<QuadNode<C, D>> {
        let num_children = 1usize << D;
        let mut children = Vec::with_capacity(num_children);
        for ci in 0..num_children {
            let mut child_min = [C::zero(); D];
            let mut child_max = [C::zero(); D];
            for d in 0..D {
                let lo: f64 = self.bounds.min.coords()[d].into();
                let hi: f64 = self.bounds.max.coords()[d].into();
                let mid = (lo + hi) * 0.5;
                if (ci >> d) & 1 == 0 {
                    child_min[d] = C::from(lo as f32);
                    child_max[d] = C::from(mid as f32);
                } else {
                    child_min[d] = C::from(mid as f32);
                    child_max[d] = C::from(hi as f32);
                }
            }
            let child_bounds = BBox::new(Point::new(child_min), Point::new(child_max));
            let mut loose_min = [C::zero(); D];
            let mut loose_max = [C::zero(); D];
            for d in 0..D {
                let lo: f64 = child_bounds.min.coords()[d].into();
                let hi: f64 = child_bounds.max.coords()[d].into();
                let half = (hi - lo) * 0.5;
                loose_min[d] = C::from((lo - half) as f32);
                loose_max[d] = C::from((hi + half) as f32);
            }
            let loose_bounds = BBox::new(Point::new(loose_min), Point::new(loose_max));
            children.push(QuadNode::new_leaf(
                child_bounds,
                loose_bounds,
                self.depth + 1,
            ));
        }
        children
    }

    fn insert(&mut self, point: Point<C, D>, id: EntryId) {
        if let Some(ref mut children) = self.children {
            let ci = self.child_index(&point);
            children[ci].insert(point, id);
            return;
        }
        self.entries.push((point, id));
        if self.entries.len() > LEAF_CAPACITY && self.depth < MAX_DEPTH {
            self.split();
        }
    }

    fn split(&mut self) {
        let mut children = self.make_children();
        let entries: Vec<(Point<C, D>, EntryId)> = self.entries.drain(..).collect();
        for (pt, id) in entries {
            let ci = self.child_index(&pt);
            children[ci].entries.push((pt, id));
        }
        self.children = Some(children);
    }

    fn remove(&mut self, id: EntryId) -> bool {
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                if child.remove(id) {
                    return true;
                }
            }
            return false;
        }
        if let Some(pos) = self.entries.iter().position(|(_, eid)| *eid == id) {
            self.entries.swap_remove(pos);
            return true;
        }
        false
    }

    fn range_query(&self, bbox: &BBox<C, D>, out: &mut Vec<EntryId>) {
        if !self.loose_bounds.intersects(bbox) {
            return;
        }
        if let Some(ref children) = self.children {
            for child in children {
                child.range_query(bbox, out);
            }
            return;
        }
        for (pt, id) in &self.entries {
            if bbox.contains_point(pt) {
                out.push(*id);
            }
        }
    }

    fn collect_all(&self, out: &mut Vec<(Point<C, D>, EntryId)>) {
        if let Some(ref children) = self.children {
            for child in children {
                child.collect_all(out);
            }
            return;
        }
        out.extend_from_slice(&self.entries);
    }

    fn node_count(&self) -> usize {
        match &self.children {
            None => 1,
            Some(children) => 1 + children.iter().map(|c| c.node_count()).sum::<usize>(),
        }
    }

    fn max_depth(&self) -> usize {
        match &self.children {
            None => self.depth,
            Some(children) => children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth),
        }
    }
}

/// Loose 2^D-tree spatial index.
///
/// Generalises the 2D quadtree (4 children), 3D octree (8 children), and 4D
/// hexadecatree (16 children) under a single generic struct parameterised by
/// `D`. The policy engine excludes this backend from migration candidates when
/// `D > 4`.
pub struct Quadtree<T, C, const D: usize> {
    root: QuadNode<C, D>,
    id_to_payload: HashMap<u64, T>,
    id_to_point: HashMap<u64, Point<C, D>>,
    len: usize,
    next_id: u64,
}

impl<T, C: CoordType, const D: usize> Quadtree<T, C, D> {
    /// Create an empty Quadtree covering the unit hypercube `[0, 1]^D`.
    pub fn new() -> Self {
        Self::with_bounds(
            Point::new([C::zero(); D]),
            Point::new([C::from(1.0_f32); D]),
        )
    }

    /// Create an empty Quadtree with explicit root bounds.
    pub fn with_bounds(min: Point<C, D>, max: Point<C, D>) -> Self {
        let bounds = BBox::new(min, max);
        let mut loose_min = [C::zero(); D];
        let mut loose_max = [C::zero(); D];
        for d in 0..D {
            let lo: f64 = min.coords()[d].into();
            let hi: f64 = max.coords()[d].into();
            let half = (hi - lo) * 0.5;
            loose_min[d] = C::from((lo - half) as f32);
            loose_max[d] = C::from((hi + half) as f32);
        }
        let loose_bounds = BBox::new(Point::new(loose_min), Point::new(loose_max));
        Self {
            root: QuadNode::new_leaf(bounds, loose_bounds, 0),
            id_to_payload: HashMap::new(),
            id_to_point: HashMap::new(),
            len: 0,
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> EntryId {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        id
    }

    fn ensure_bounds(&mut self, point: &Point<C, D>) {
        let mut needs_expand = false;
        for d in 0..D {
            let v: f64 = point.coords()[d].into();
            let lo: f64 = self.root.bounds.min.coords()[d].into();
            let hi: f64 = self.root.bounds.max.coords()[d].into();
            if v < lo || v > hi {
                needs_expand = true;
                break;
            }
        }
        if !needs_expand {
            return;
        }
        let mut all: Vec<(Point<C, D>, EntryId)> = Vec::with_capacity(self.len);
        self.root.collect_all(&mut all);

        let mut new_min = [C::zero(); D];
        let mut new_max = [C::zero(); D];
        for d in 0..D {
            let lo: f64 = self.root.bounds.min.coords()[d].into();
            let hi: f64 = self.root.bounds.max.coords()[d].into();
            let v: f64 = point.coords()[d].into();
            let span = (hi - lo).max(1.0);
            let new_lo = lo.min(v - span * 0.1);
            let new_hi = hi.max(v + span * 0.1);
            new_min[d] = C::from(new_lo as f32);
            new_max[d] = C::from(new_hi as f32);
        }

        let mut new_tree = Quadtree::with_bounds(Point::new(new_min), Point::new(new_max));
        new_tree.next_id = self.next_id;
        for (pt, id) in all {
            new_tree.root.insert(pt, id);
            new_tree.id_to_point.insert(id.0, pt);
        }
        new_tree.id_to_payload = std::mem::take(&mut self.id_to_payload);
        new_tree.len = self.len;
        *self = new_tree;
    }

    pub fn insert_entry(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        self.ensure_bounds(&point);
        let id = self.alloc_id();
        self.id_to_point.insert(id.0, point);
        self.id_to_payload.insert(id.0, payload);
        self.root.insert(point, id);
        self.len += 1;
        id
    }

    pub fn remove_entry(&mut self, id: EntryId) -> Option<T> {
        self.id_to_point.remove(&id.0)?;
        let payload = self.id_to_payload.remove(&id.0)?;
        self.root.remove(id);
        self.len -= 1;
        Some(payload)
    }

    pub fn range_query_impl<'a>(&'a self, bbox: &BBox<C, D>) -> Vec<(EntryId, &'a T)> {
        let mut ids = Vec::new();
        self.root.range_query(bbox, &mut ids);
        ids.into_iter()
            .filter_map(|id| self.id_to_payload.get(&id.0).map(|p| (id, p)))
            .collect()
    }

    pub fn knn_query_impl<'a>(
        &'a self,
        point: &Point<C, D>,
        k: usize,
    ) -> Vec<(f64, EntryId, &'a T)> {
        if k == 0 {
            return Vec::new();
        }
        let mut all: Vec<(Point<C, D>, EntryId)> = Vec::with_capacity(self.len);
        self.root.collect_all(&mut all);
        let mut results: Vec<(f64, EntryId, &'a T)> = all
            .into_iter()
            .filter_map(|(p, id)| {
                self.id_to_payload.get(&id.0).map(|payload| {
                    let dist = point_dist_sq(&p, point).sqrt();
                    (dist, id, payload)
                })
            })
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn collect_all_with_payload(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        let mut raw: Vec<(Point<C, D>, EntryId)> = Vec::with_capacity(self.len);
        self.root.collect_all(&mut raw);
        raw.into_iter()
            .filter_map(|(p, id)| {
                self.id_to_payload
                    .get(&id.0)
                    .map(|payload| (p, id, payload))
            })
            .collect()
    }

    /// Return the total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Return the maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }
}

fn point_dist_sq<C: CoordType, const D: usize>(a: &Point<C, D>, b: &Point<C, D>) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let da: f64 = a.coords()[d].into();
        let db: f64 = b.coords()[d].into();
        let diff = da - db;
        sum += diff * diff;
    }
    sum
}

impl<T, C: CoordType, const D: usize> Default for Quadtree<T, C, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static, C: CoordType, const D: usize> SpatialBackend<T, C, D>
    for Quadtree<T, C, D>
{
    fn insert(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        self.insert_entry(point, payload)
    }

    fn remove(&mut self, id: EntryId) -> Option<T> {
        self.remove_entry(id)
    }

    fn range_query(&self, bbox: &BBox<C, D>) -> Vec<(EntryId, &T)> {
        self.range_query_impl(bbox)
    }

    fn knn_query(&self, point: &Point<C, D>, k: usize) -> Vec<(f64, EntryId, &T)> {
        self.knn_query_impl(point, k)
    }

    fn spatial_join(&self, other: &dyn SpatialBackend<T, C, D>) -> Vec<(EntryId, EntryId)> {
        let self_entries = self.collect_all_with_payload();
        let other_entries = other.all_entries();
        let mut pairs = Vec::new();
        for (pa, id_a, _) in &self_entries {
            let bbox_a = BBox::new(*pa, *pa);
            for (pb, id_b, _) in &other_entries {
                if bbox_a.intersects(&BBox::new(*pb, *pb)) {
                    pairs.push((*id_a, *id_b));
                }
            }
        }
        pairs
    }

    fn bulk_load(entries: Vec<(Point<C, D>, T)>) -> Self {
        if entries.is_empty() {
            return Self::new();
        }
        let mut min_c = *entries[0].0.coords();
        let mut max_c = *entries[0].0.coords();
        for (p, _) in &entries {
            for d in 0..D {
                let v = p.coords()[d];
                if v < min_c[d] {
                    min_c[d] = v;
                }
                if v > max_c[d] {
                    max_c[d] = v;
                }
            }
        }
        let mut padded_min = [C::zero(); D];
        let mut padded_max = [C::zero(); D];
        for d in 0..D {
            let lo: f64 = min_c[d].into();
            let hi: f64 = max_c[d].into();
            let margin = ((hi - lo) * 0.01).max(1.0);
            padded_min[d] = C::from((lo - margin) as f32);
            padded_max[d] = C::from((hi + margin) as f32);
        }
        let mut tree = Self::with_bounds(Point::new(padded_min), Point::new(padded_max));
        for (point, payload) in entries {
            tree.insert_entry(point, payload);
        }
        tree
    }

    fn len(&self) -> usize {
        self.len
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Quadtree
    }

    fn all_entries(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        self.collect_all_with_payload()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn rand_pts<const D: usize>(n: usize, seed: u64) -> Vec<Point<f64, D>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let coords: [f64; D] = std::array::from_fn(|_| rng.next_f64() * 1000.0);
                Point::new(coords)
            })
            .collect()
    }

    fn brute_range<C: CoordType, const D: usize>(
        pts: &[(Point<C, D>, EntryId)],
        bbox: &BBox<C, D>,
    ) -> Vec<EntryId> {
        let mut ids: Vec<EntryId> = pts
            .iter()
            .filter(|(p, _)| bbox.contains_point(p))
            .map(|(_, id)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    #[test]
    fn children_count_d2() {
        let pts = rand_pts::<2>(200, 1);
        let mut tree = Quadtree::<usize, f64, 2>::bulk_load(
            pts.into_iter().enumerate().map(|(i, p)| (p, i)).collect(),
        );
        for i in 200..300usize {
            tree.insert(rand_pts::<2>(1, i as u64 + 9999)[0], i);
        }
        assert!(tree.root.children.is_some());
        assert_eq!(tree.root.children.as_ref().unwrap().len(), 1 << 2);
    }

    #[test]
    fn children_count_d3() {
        let pts = rand_pts::<3>(200, 2);
        let mut tree = Quadtree::<usize, f64, 3>::bulk_load(
            pts.into_iter().enumerate().map(|(i, p)| (p, i)).collect(),
        );
        for i in 200..300usize {
            tree.insert(rand_pts::<3>(1, i as u64 + 8888)[0], i);
        }
        assert!(tree.root.children.is_some());
        assert_eq!(tree.root.children.as_ref().unwrap().len(), 1 << 3);
    }

    #[test]
    fn children_count_d4() {
        let pts = rand_pts::<4>(200, 3);
        let mut tree = Quadtree::<usize, f64, 4>::bulk_load(
            pts.into_iter().enumerate().map(|(i, p)| (p, i)).collect(),
        );
        for i in 200..300usize {
            tree.insert(rand_pts::<4>(1, i as u64 + 7777)[0], i);
        }
        assert!(tree.root.children.is_some());
        assert_eq!(tree.root.children.as_ref().unwrap().len(), 1 << 4);
    }

    #[test]
    fn each_point_assigned_to_exactly_one_child_d2() {
        let pts = rand_pts::<2>(300, 42);
        let tree = Quadtree::<usize, f64, 2>::bulk_load(
            pts.iter()
                .cloned()
                .enumerate()
                .map(|(i, p)| (p, i))
                .collect(),
        );
        if let Some(ref children) = tree.root.children {
            let mut all_ids: Vec<EntryId> = Vec::new();
            for child in children {
                let mut child_entries: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
                child.collect_all(&mut child_entries);
                all_ids.extend(child_entries.iter().map(|(_, id)| *id));
            }
            all_ids.sort_by_key(|id| id.0);
            let before = all_ids.len();
            all_ids.dedup();
            assert_eq!(before, all_ids.len());
        }
    }

    #[test]
    fn range_query_basic() {
        let mut tree = Quadtree::<u32, f64, 2>::new();
        let id1 = tree.insert(Point::new([0.5, 0.5]), 1u32);
        let id2 = tree.insert(Point::new([1.5, 1.5]), 2u32);
        let _id3 = tree.insert(Point::new([5.0, 5.0]), 3u32);
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([2.0, 2.0]));
        let mut got: Vec<EntryId> = tree
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        assert_eq!(got, vec![id1, id2]);
    }

    #[test]
    fn remove_works() {
        let mut tree = Quadtree::<u32, f64, 2>::new();
        let id1 = tree.insert(Point::new([1.0, 1.0]), 10u32);
        let id2 = tree.insert(Point::new([2.0, 2.0]), 20u32);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.remove(id1), Some(10u32));
        assert_eq!(tree.len(), 1);
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([3.0, 3.0]));
        let ids: Vec<EntryId> = tree
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(!ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn kind_is_quadtree() {
        assert_eq!(Quadtree::<u32, f64, 2>::new().kind(), BackendKind::Quadtree);
    }

    #[test]
    fn range_query_vs_brute_force_2d() {
        let pts = rand_pts::<2>(1000, 99);
        let mut tree = Quadtree::<usize, f64, 2>::new();
        let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = tree.insert(p, i);
            pt_ids.push((p, id));
        }
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([500.0, 500.0]));
        let mut got: Vec<EntryId> = tree
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        let expected = brute_range(&pt_ids, &bbox);
        assert_eq!(got, expected);
    }

    fn pt2d() -> impl Strategy<Value = Point<f64, 2>> {
        (0.0_f64..1000.0, 0.0_f64..1000.0).prop_map(|(x, y)| Point::new([x, y]))
    }

    fn bbox2d() -> impl Strategy<Value = BBox<f64, 2>> {
        (
            0.0_f64..900.0,
            0.0_f64..900.0,
            10.0_f64..200.0,
            10.0_f64..200.0,
        )
            .prop_map(|(x, y, w, h)| BBox::new(Point::new([x, y]), Point::new([x + w, y + h])))
    }

    // A split node must have exactly 2^D children, and every point must appear
    // in exactly one child.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_quadtree_children_count_d2(
            pts in prop::collection::vec(pt2d(), (LEAF_CAPACITY + 1)..50),
        ) {
            let mut tree = Quadtree::<usize, f64, 2>::new();
            for (i, p) in pts.iter().enumerate() {
                tree.insert(*p, i);
            }
            if let Some(ref children) = tree.root.children {
                prop_assert_eq!(children.len(), 1 << 2usize);
                for child in children {
                    if let Some(ref gc) = child.children {
                        prop_assert_eq!(gc.len(), 1 << 2usize);
                    }
                }
            }
            let mut raw: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            tree.root.collect_all(&mut raw);
            let mut all_ids: Vec<EntryId> = raw.iter().map(|(_, id)| *id).collect();
            all_ids.sort_by_key(|id| id.0);
            let before = all_ids.len();
            all_ids.dedup();
            prop_assert_eq!(before, all_ids.len());
            prop_assert_eq!(all_ids.len(), pts.len());
        }
    }

    // Insert-Remove Round Trip
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_insert_remove_round_trip_quadtree(
            pts in prop::collection::vec(pt2d(), 1..50),
            remove_indices in prop::collection::vec(0usize..50, 0..25),
        ) {
            let mut tree = Quadtree::<usize, f64, 2>::new();
            let mut inserted: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, &p) in pts.iter().enumerate() {
                let id = tree.insert(p, i);
                inserted.push((p, id));
            }
            let mut removed_ids: Vec<EntryId> = Vec::new();
            for &ri in &remove_indices {
                let idx = ri % inserted.len();
                let (_, id) = inserted[idx];
                if !removed_ids.contains(&id) {
                    let result = tree.remove(id);
                    prop_assert!(result.is_some(), "remove returned None for inserted id");
                    removed_ids.push(id);
                }
            }
            let full_bbox = BBox::new(Point::new([-1.0e9, -1.0e9]), Point::new([1.0e9, 1.0e9]));
            let remaining_ids: Vec<EntryId> = tree.range_query(&full_bbox)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for &removed_id in &removed_ids {
                prop_assert!(
                    !remaining_ids.contains(&removed_id),
                    "removed entry {:?} still appears in range query",
                    removed_id
                );
            }
            let expected_len = inserted.len() - removed_ids.len();
            prop_assert_eq!(tree.len(), expected_len);
        }
    }

    // For any dataset and bbox, range_query result must equal brute-force scan.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_range_query_oracle_quadtree(
            pts in prop::collection::vec(pt2d(), 1..100),
            bbox in bbox2d(),
        ) {
            let mut tree = Quadtree::<usize, f64, 2>::new();
            let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                let id = tree.insert(*p, i);
                pt_ids.push((*p, id));
            }
            let mut got: Vec<EntryId> =
                tree.range_query(&bbox).into_iter().map(|(id, _)| id).collect();
            got.sort_by_key(|id| id.0);
            let expected = brute_range(&pt_ids, &bbox);
            prop_assert_eq!(got, expected);
        }
    }
}
