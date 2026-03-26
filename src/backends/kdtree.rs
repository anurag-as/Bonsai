//! KD-tree spatial index backend with sliding midpoint splits.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::backends::SpatialBackend;
use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

const LEAF_CAPACITY: usize = 8;

enum KDNode<C, const D: usize> {
    Leaf {
        entries: Vec<(Point<C, D>, EntryId)>,
    },
    Split {
        axis: usize,
        split_val: C,
        left: Box<KDNode<C, D>>,
        right: Box<KDNode<C, D>>,
    },
}

fn entries_bbox<C: CoordType, const D: usize>(entries: &[(Point<C, D>, EntryId)]) -> BBox<C, D> {
    debug_assert!(!entries.is_empty());
    let mut min_c = *entries[0].0.coords();
    let mut max_c = *entries[0].0.coords();
    for (p, _) in entries.iter().skip(1) {
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
    BBox::new(Point::new(min_c), Point::new(max_c))
}

fn build_node<C: CoordType, const D: usize>(
    mut entries: Vec<(Point<C, D>, EntryId)>,
    bbox: BBox<C, D>,
) -> KDNode<C, D> {
    if entries.len() <= LEAF_CAPACITY {
        return KDNode::Leaf { entries };
    }

    // Choose the axis with the largest spread.
    let mut best_axis = 0;
    let mut best_span = C::zero();
    for d in 0..D {
        let span = bbox.max.coords()[d] - bbox.min.coords()[d];
        if span > best_span {
            best_span = span;
            best_axis = d;
        }
    }

    if best_span == C::zero() {
        return KDNode::Leaf { entries };
    }

    // Median split guarantees ceil(log2(n)) + 1 depth.
    entries.sort_by(|(a, _), (b, _)| {
        let av: f64 = a.coords()[best_axis].into();
        let bv: f64 = b.coords()[best_axis].into();
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid_idx = entries.len() / 2;
    let right_entries = entries.split_off(mid_idx);
    let left_entries = entries;

    let actual_split_f64 = left_entries
        .iter()
        .map(|(p, _)| {
            let v: f64 = p.coords()[best_axis].into();
            v
        })
        .fold(f64::NEG_INFINITY, f64::max);

    let split_val: C = C::from(actual_split_f64 as f32);

    let mut left_bbox = bbox;
    left_bbox.max.coords_mut()[best_axis] = split_val;

    let mut right_bbox = bbox;
    right_bbox.min.coords_mut()[best_axis] = split_val;

    let left = Box::new(build_node(left_entries, left_bbox));
    let right = Box::new(build_node(right_entries, right_bbox));

    KDNode::Split {
        axis: best_axis,
        split_val,
        left,
        right,
    }
}

fn node_range<C: CoordType, const D: usize>(
    node: &KDNode<C, D>,
    bbox: &BBox<C, D>,
    node_bbox: &BBox<C, D>,
    out: &mut Vec<EntryId>,
) {
    if !node_bbox.intersects(bbox) {
        return;
    }
    match node {
        KDNode::Leaf { entries } => {
            for (p, id) in entries {
                if bbox.contains_point(p) {
                    out.push(*id);
                }
            }
        }
        KDNode::Split {
            axis,
            split_val,
            left,
            right,
        } => {
            let mut left_bbox = *node_bbox;
            left_bbox.max.coords_mut()[*axis] = *split_val;
            let mut right_bbox = *node_bbox;
            right_bbox.min.coords_mut()[*axis] = *split_val;

            node_range(left, bbox, &left_bbox, out);
            node_range(right, bbox, &right_bbox, out);
        }
    }
}

fn node_collect<C: CoordType, const D: usize>(
    node: &KDNode<C, D>,
    out: &mut Vec<(Point<C, D>, EntryId)>,
) {
    match node {
        KDNode::Leaf { entries } => {
            out.extend_from_slice(entries);
        }
        KDNode::Split { left, right, .. } => {
            node_collect(left, out);
            node_collect(right, out);
        }
    }
}

fn node_depth<C, const D: usize>(node: &KDNode<C, D>) -> usize {
    match node {
        KDNode::Leaf { .. } => 1,
        KDNode::Split { left, right, .. } => 1 + node_depth(left).max(node_depth(right)),
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

/// KD-tree spatial index with sliding midpoint splits.
///
/// Guarantees O(log n) depth for any data distribution via median splits.
pub struct KDTree<T, C, const D: usize> {
    tree: Mutex<(KDNode<C, D>, BBox<C, D>)>,
    flat: Vec<(Point<C, D>, EntryId)>,
    id_to_payload: HashMap<u64, T>,
    id_to_point: HashMap<u64, Point<C, D>>,
    len: usize,
    next_id: u64,
    dirty: Mutex<bool>,
}

impl<T, C: CoordType, const D: usize> KDTree<T, C, D> {
    pub fn new() -> Self {
        let empty_bbox = BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D]));
        Self {
            tree: Mutex::new((
                KDNode::Leaf {
                    entries: Vec::new(),
                },
                empty_bbox,
            )),
            flat: Vec::new(),
            id_to_payload: HashMap::new(),
            id_to_point: HashMap::new(),
            len: 0,
            next_id: 0,
            dirty: Mutex::new(false),
        }
    }

    fn alloc_id(&mut self) -> EntryId {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        id
    }

    fn rebuild(&self) {
        let empty_bbox = BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D]));
        if self.flat.is_empty() {
            *self.tree.lock().unwrap() = (
                KDNode::Leaf {
                    entries: Vec::new(),
                },
                empty_bbox,
            );
            *self.dirty.lock().unwrap() = false;
            return;
        }
        let bbox = entries_bbox(&self.flat);
        let root = build_node(self.flat.clone(), bbox);
        *self.tree.lock().unwrap() = (root, bbox);
        *self.dirty.lock().unwrap() = false;
    }

    fn ensure_built(&self) {
        if *self.dirty.lock().unwrap() {
            self.rebuild();
        }
    }

    /// Return the current depth of the tree.
    pub fn depth(&self) -> usize {
        self.ensure_built();
        node_depth(&self.tree.lock().unwrap().0)
    }

    pub fn insert_entry(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        let id = self.alloc_id();
        self.id_to_point.insert(id.0, point);
        self.id_to_payload.insert(id.0, payload);
        self.flat.push((point, id));
        self.len += 1;
        *self.dirty.lock().unwrap() = true;
        id
    }

    pub fn remove_entry(&mut self, id: EntryId) -> Option<T> {
        self.id_to_point.remove(&id.0)?;
        let payload = self.id_to_payload.remove(&id.0)?;
        if let Some(pos) = self.flat.iter().position(|(_, eid)| *eid == id) {
            self.flat.swap_remove(pos);
        }
        self.len -= 1;
        *self.dirty.lock().unwrap() = true;
        Some(payload)
    }

    pub fn range_query_impl<'a>(&'a self, bbox: &BBox<C, D>) -> Vec<(EntryId, &'a T)> {
        self.ensure_built();
        if self.len == 0 {
            return Vec::new();
        }
        let tree = self.tree.lock().unwrap();
        let (root, root_bbox) = &*tree;
        let mut ids = Vec::new();
        node_range(root, bbox, root_bbox, &mut ids);
        drop(tree);
        ids.into_iter()
            .filter_map(|id| self.id_to_payload.get(&id.0).map(|p| (id, p)))
            .collect()
    }

    pub fn knn_query_impl<'a>(
        &'a self,
        point: &Point<C, D>,
        k: usize,
    ) -> Vec<(f64, EntryId, &'a T)> {
        self.ensure_built();
        if k == 0 {
            return Vec::new();
        }
        let tree = self.tree.lock().unwrap();
        let (root, _) = &*tree;
        let mut raw: Vec<(Point<C, D>, EntryId)> = Vec::new();
        node_collect(root, &mut raw);
        drop(tree);
        let mut all: Vec<(f64, EntryId, &'a T)> = raw
            .into_iter()
            .filter_map(|(p, id)| {
                self.id_to_payload
                    .get(&id.0)
                    .map(|payload| (point_dist_sq(&p, point).sqrt(), id, payload))
            })
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    fn collect_all(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        self.ensure_built();
        let tree = self.tree.lock().unwrap();
        let (root, _) = &*tree;
        let mut raw: Vec<(Point<C, D>, EntryId)> = Vec::new();
        node_collect(root, &mut raw);
        drop(tree);
        raw.into_iter()
            .filter_map(|(p, id)| {
                self.id_to_payload
                    .get(&id.0)
                    .map(|payload| (p, id, payload))
            })
            .collect()
    }
}

impl<T, C: CoordType, const D: usize> Default for KDTree<T, C, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static, C: CoordType, const D: usize> SpatialBackend<T, C, D>
    for KDTree<T, C, D>
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
        let self_entries = self.collect_all();
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
        let mut tree = Self::new();
        for (point, payload) in entries {
            tree.insert_entry(point, payload);
        }
        tree
    }

    fn len(&self) -> usize {
        self.len
    }

    fn kind(&self) -> BackendKind {
        BackendKind::KDTree
    }

    fn all_entries(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        self.collect_all()
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
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn rand_pts_2d(n: usize, seed: u64) -> Vec<Point<f64, 2>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
            .collect()
    }

    fn clustered_pts_2d(n: usize, clusters: usize, seed: u64) -> Vec<Point<f64, 2>> {
        let mut rng = Lcg::new(seed);
        let centres: Vec<[f64; 2]> = (0..clusters)
            .map(|_| {
                [
                    rng.next_f64() * 800.0 + 100.0,
                    rng.next_f64() * 800.0 + 100.0,
                ]
            })
            .collect();
        (0..n)
            .map(|i| {
                let c = &centres[i % clusters];
                let x = c[0] + rng.next_normal() * 20.0;
                let y = c[1] + rng.next_normal() * 20.0;
                Point::new([x.clamp(0.0, 1000.0), y.clamp(0.0, 1000.0)])
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
    fn insert_and_len() {
        let mut tree = KDTree::<u32, f64, 2>::new();
        assert_eq!(tree.len(), 0);
        tree.insert(Point::new([0.5, 0.5]), 1u32);
        assert_eq!(tree.len(), 1);
        tree.insert(Point::new([1.5, 1.5]), 2u32);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn range_query_basic() {
        let mut tree = KDTree::<u32, f64, 2>::new();
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
        let mut tree = KDTree::<u32, f64, 2>::new();
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
    fn kind_is_kdtree() {
        assert_eq!(KDTree::<u32, f64, 2>::new().kind(), BackendKind::KDTree);
    }

    #[test]
    fn range_query_vs_brute_force_2d_10k() {
        let pts = rand_pts_2d(10_000, 42);
        let mut tree = KDTree::<usize, f64, 2>::new();
        let mut pt_ids = Vec::new();
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

    fn check_depth_bound<const D: usize>(n: usize, seed: u64)
    where
        [(); D]:,
    {
        let mut rng = Lcg::new(seed);
        let mut tree = KDTree::<usize, f64, D>::new();
        for i in 0..n {
            let coords: [f64; D] = std::array::from_fn(|_| rng.next_f64() * 1000.0);
            tree.insert(Point::new(coords), i);
        }
        let depth = tree.depth();
        let bound = (n as f64).log2().ceil() as usize + 1;
        assert!(
            depth <= bound,
            "D={D}: depth {depth} exceeds bound {bound} for n={n}"
        );
    }

    #[test]
    fn depth_bound_d2() {
        check_depth_bound::<2>(1000, 1);
    }
    #[test]
    fn depth_bound_d3() {
        check_depth_bound::<3>(1000, 2);
    }
    #[test]
    fn depth_bound_d4() {
        check_depth_bound::<4>(1000, 3);
    }
    #[test]
    fn depth_bound_d5() {
        check_depth_bound::<5>(1000, 4);
    }
    #[test]
    fn depth_bound_d6() {
        check_depth_bound::<6>(1000, 5);
    }
    #[test]
    fn depth_bound_d7() {
        check_depth_bound::<7>(1000, 6);
    }
    #[test]
    fn depth_bound_d8() {
        check_depth_bound::<8>(1000, 7);
    }

    #[test]
    fn depth_bound_clustered_2d() {
        let pts = clustered_pts_2d(5000, 20, 99);
        let mut tree = KDTree::<usize, f64, 2>::new();
        for (i, &p) in pts.iter().enumerate() {
            tree.insert(p, i);
        }
        let n = pts.len();
        let depth = tree.depth();
        let bound = (n as f64).log2().ceil() as usize + 1;
        assert!(
            depth <= bound,
            "clustered: depth {depth} exceeds bound {bound} for n={n}"
        );
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

    // KD-tree depth must not exceed ceil(log2(n)) + 1 for any dataset.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_kdtree_depth_bound(pts in prop::collection::vec(pt2d(), 2..200)) {
            let mut tree = KDTree::<usize, f64, 2>::new();
            for (i, p) in pts.iter().enumerate() {
                tree.insert(*p, i);
            }
            let n = pts.len();
            let depth = tree.depth();
            let bound = (n as f64).log2().ceil() as usize + 1;
            prop_assert!(depth <= bound, "depth {} exceeds bound {} for n={}", depth, bound, n);
        }
    }

    // Range query result must equal brute-force linear scan for any dataset and bbox.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_range_query_oracle_kdtree(
            pts in prop::collection::vec(pt2d(), 1..100),
            bbox in bbox2d(),
        ) {
            let mut tree = KDTree::<usize, f64, 2>::new();
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
