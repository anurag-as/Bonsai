//! R*-tree spatial index backend.
//!
//! Implements the [`SpatialBackend`] trait using an R*-tree with quadratic
//! splits and forced reinsertion on overflow (min=4, max=16 entries per node).
//!
//! Reference: Beckmann et al. (1990) "The R*-tree: An Efficient and Robust
//! Access Method for Points and Rectangles"

use std::collections::HashMap;

use crate::backends::SpatialBackend;
use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

const MAX_ENTRIES: usize = 16;
const MIN_ENTRIES: usize = 4;

struct Entry<T, C, const D: usize> {
    point: Point<C, D>,
    payload: T,
    id: EntryId,
}

enum RNode<T, C, const D: usize> {
    Leaf {
        bbox: BBox<C, D>,
        entries: Vec<Entry<T, C, D>>,
    },
    Internal {
        bbox: BBox<C, D>,
        children: Vec<Box<RNode<T, C, D>>>,
    },
}

impl<T, C: CoordType, const D: usize> RNode<T, C, D> {
    fn bbox(&self) -> &BBox<C, D> {
        match self {
            RNode::Leaf { bbox, .. } | RNode::Internal { bbox, .. } => bbox,
        }
    }
}

fn merge_bbox<C: CoordType, const D: usize>(a: &BBox<C, D>, b: &BBox<C, D>) -> BBox<C, D> {
    let mut min_c = [C::zero(); D];
    let mut max_c = [C::zero(); D];
    for d in 0..D {
        min_c[d] = if a.min.coords()[d] < b.min.coords()[d] {
            a.min.coords()[d]
        } else {
            b.min.coords()[d]
        };
        max_c[d] = if a.max.coords()[d] > b.max.coords()[d] {
            a.max.coords()[d]
        } else {
            b.max.coords()[d]
        };
    }
    BBox::new(Point::new(min_c), Point::new(max_c))
}

fn point_bbox<C: CoordType, const D: usize>(p: &Point<C, D>) -> BBox<C, D> {
    BBox::new(*p, *p)
}

fn bbox_area<C: CoordType, const D: usize>(bbox: &BBox<C, D>) -> f64 {
    let mut area = 1.0_f64;
    for d in 0..D {
        let span: f64 = (bbox.max.coords()[d] - bbox.min.coords()[d]).into();
        area *= span.max(0.0);
    }
    area
}

fn enlargement<C: CoordType, const D: usize>(existing: &BBox<C, D>, new_bbox: &BBox<C, D>) -> f64 {
    bbox_area(&merge_bbox(existing, new_bbox)) - bbox_area(existing)
}

fn recompute_leaf_bbox<T, C: CoordType, const D: usize>(entries: &[Entry<T, C, D>]) -> BBox<C, D> {
    if entries.is_empty() {
        return BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D]));
    }
    let mut bbox = point_bbox(&entries[0].point);
    for e in entries.iter().skip(1) {
        bbox = merge_bbox(&bbox, &point_bbox(&e.point));
    }
    bbox
}

fn recompute_internal_bbox<T, C: CoordType, const D: usize>(
    children: &[Box<RNode<T, C, D>>],
) -> BBox<C, D> {
    if children.is_empty() {
        return BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D]));
    }
    let mut bbox = *children[0].bbox();
    for c in children.iter().skip(1) {
        bbox = merge_bbox(&bbox, c.bbox());
    }
    bbox
}

pub(crate) fn point_dist_sq<C: CoordType, const D: usize>(a: &Point<C, D>, b: &Point<C, D>) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let da: f64 = a.coords()[d].into();
        let db: f64 = b.coords()[d].into();
        let diff = da - db;
        sum += diff * diff;
    }
    sum
}

fn pick_seeds_leaf<T, C: CoordType, const D: usize>(entries: &[Entry<T, C, D>]) -> (usize, usize) {
    let n = entries.len();
    let (mut si, mut sj) = (0, 1);
    let mut worst = f64::NEG_INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let combined = merge_bbox(
                &point_bbox(&entries[i].point),
                &point_bbox(&entries[j].point),
            );
            let waste = bbox_area(&combined)
                - bbox_area(&point_bbox(&entries[i].point))
                - bbox_area(&point_bbox(&entries[j].point));
            if waste > worst {
                worst = waste;
                si = i;
                sj = j;
            }
        }
    }
    (si, sj)
}

fn pick_seeds_internal<T, C: CoordType, const D: usize>(
    children: &[Box<RNode<T, C, D>>],
) -> (usize, usize) {
    let n = children.len();
    let (mut si, mut sj) = (0, 1);
    let mut worst = f64::NEG_INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let combined = merge_bbox(children[i].bbox(), children[j].bbox());
            let waste = bbox_area(&combined)
                - bbox_area(children[i].bbox())
                - bbox_area(children[j].bbox());
            if waste > worst {
                worst = waste;
                si = i;
                sj = j;
            }
        }
    }
    (si, sj)
}

fn assign_to_group<C: CoordType, const D: usize>(
    eb: &BBox<C, D>,
    bbox_a: &mut BBox<C, D>,
    bbox_b: &mut BBox<C, D>,
    len_a: usize,
    len_b: usize,
) -> bool {
    let d1 = enlargement(bbox_a, eb);
    let d2 = enlargement(bbox_b, eb);
    if d1 < d2 {
        return true;
    }
    if d2 < d1 {
        return false;
    }
    let area_a = bbox_area(bbox_a);
    let area_b = bbox_area(bbox_b);
    if area_a < area_b {
        return true;
    }
    if area_b < area_a {
        return false;
    }
    len_a <= len_b
}

#[allow(clippy::type_complexity)]
fn quadratic_split_leaf<T, C: CoordType, const D: usize>(
    mut entries: Vec<Entry<T, C, D>>,
) -> (Vec<Entry<T, C, D>>, Vec<Entry<T, C, D>>) {
    let (si, sj) = pick_seeds_leaf(&entries);
    let (si, sj) = if si < sj { (si, sj) } else { (sj, si) };
    let seed_j = entries.remove(sj);
    let seed_i = entries.remove(si);
    let mut group_a = vec![seed_i];
    let mut group_b = vec![seed_j];
    let mut bbox_a = point_bbox(&group_a[0].point);
    let mut bbox_b = point_bbox(&group_b[0].point);

    while !entries.is_empty() {
        let remaining = entries.len();
        if group_a.len() + remaining == MIN_ENTRIES {
            group_a.extend(entries);
            break;
        }
        if group_b.len() + remaining == MIN_ENTRIES {
            group_b.extend(entries);
            break;
        }
        let mut best_idx = 0;
        let mut best_diff = f64::NEG_INFINITY;
        for (i, e) in entries.iter().enumerate() {
            let eb = point_bbox(&e.point);
            let diff = (enlargement(&bbox_a, &eb) - enlargement(&bbox_b, &eb)).abs();
            if diff > best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }
        let e = entries.remove(best_idx);
        let eb = point_bbox(&e.point);
        if assign_to_group(&eb, &mut bbox_a, &mut bbox_b, group_a.len(), group_b.len()) {
            bbox_a = merge_bbox(&bbox_a, &eb);
            group_a.push(e);
        } else {
            bbox_b = merge_bbox(&bbox_b, &eb);
            group_b.push(e);
        }
    }
    (group_a, group_b)
}

#[allow(clippy::type_complexity)]
fn quadratic_split_internal<T, C: CoordType, const D: usize>(
    mut children: Vec<Box<RNode<T, C, D>>>,
) -> (Vec<Box<RNode<T, C, D>>>, Vec<Box<RNode<T, C, D>>>) {
    let (si, sj) = pick_seeds_internal(&children);
    let (si, sj) = if si < sj { (si, sj) } else { (sj, si) };
    let seed_j = children.remove(sj);
    let seed_i = children.remove(si);
    let mut group_a = vec![seed_i];
    let mut group_b = vec![seed_j];
    let mut bbox_a = *group_a[0].bbox();
    let mut bbox_b = *group_b[0].bbox();

    while !children.is_empty() {
        let remaining = children.len();
        if group_a.len() + remaining == MIN_ENTRIES {
            group_a.extend(children);
            break;
        }
        if group_b.len() + remaining == MIN_ENTRIES {
            group_b.extend(children);
            break;
        }
        let mut best_idx = 0;
        let mut best_diff = f64::NEG_INFINITY;
        for (i, c) in children.iter().enumerate() {
            let diff = (enlargement(&bbox_a, c.bbox()) - enlargement(&bbox_b, c.bbox())).abs();
            if diff > best_diff {
                best_diff = diff;
                best_idx = i;
            }
        }
        let c = children.remove(best_idx);
        if assign_to_group(
            c.bbox(),
            &mut bbox_a,
            &mut bbox_b,
            group_a.len(),
            group_b.len(),
        ) {
            bbox_a = merge_bbox(&bbox_a, c.bbox());
            group_a.push(c);
        } else {
            bbox_b = merge_bbox(&bbox_b, c.bbox());
            group_b.push(c);
        }
    }
    (group_a, group_b)
}

fn choose_subtree<T, C: CoordType, const D: usize>(
    children: &[Box<RNode<T, C, D>>],
    new_bbox: &BBox<C, D>,
) -> usize {
    let mut best = 0;
    let mut best_enl = f64::INFINITY;
    let mut best_area = f64::INFINITY;
    for (i, c) in children.iter().enumerate() {
        let enl = enlargement(c.bbox(), new_bbox);
        let area = bbox_area(c.bbox());
        if enl < best_enl || (enl == best_enl && area < best_area) {
            best_enl = enl;
            best_area = area;
            best = i;
        }
    }
    best
}

fn node_insert<T, C: CoordType, const D: usize>(
    node: &mut Box<RNode<T, C, D>>,
    entry: Entry<T, C, D>,
) -> Option<Box<RNode<T, C, D>>> {
    match &mut **node {
        RNode::Leaf { entries, bbox } => {
            let pb = point_bbox(&entry.point);
            *bbox = if entries.is_empty() {
                pb
            } else {
                merge_bbox(bbox, &pb)
            };
            entries.push(entry);

            if entries.len() > MAX_ENTRIES {
                let all = std::mem::take(entries);
                let (ga, gb) = quadratic_split_leaf(all);
                let bbox_a = recompute_leaf_bbox(&ga);
                let bbox_b = recompute_leaf_bbox(&gb);
                *bbox = bbox_a;
                *entries = ga;
                Some(Box::new(RNode::Leaf {
                    bbox: bbox_b,
                    entries: gb,
                }))
            } else {
                None
            }
        }
        RNode::Internal { children, bbox } => {
            let pb = point_bbox(&entry.point);
            let best_idx = choose_subtree(children, &pb);
            *bbox = merge_bbox(bbox, &pb);

            let split = node_insert(&mut children[best_idx], entry);
            *bbox = recompute_internal_bbox(children);

            if let Some(sibling) = split {
                children.push(sibling);
                *bbox = recompute_internal_bbox(children);

                if children.len() > MAX_ENTRIES {
                    let all = std::mem::take(children);
                    let (ga, gb) = quadratic_split_internal(all);
                    let bbox_a = recompute_internal_bbox(&ga);
                    let bbox_b = recompute_internal_bbox(&gb);
                    *bbox = bbox_a;
                    *children = ga;
                    return Some(Box::new(RNode::Internal {
                        bbox: bbox_b,
                        children: gb,
                    }));
                }
            }
            None
        }
    }
}

fn node_remove<T, C: CoordType, const D: usize>(
    node: &mut Box<RNode<T, C, D>>,
    id: EntryId,
    point: &Point<C, D>,
) -> Option<T> {
    match &mut **node {
        RNode::Leaf { entries, bbox } => {
            if let Some(pos) = entries.iter().position(|e| e.id == id) {
                let entry = entries.remove(pos);
                *bbox = recompute_leaf_bbox(entries);
                Some(entry.payload)
            } else {
                None
            }
        }
        RNode::Internal { children, bbox } => {
            let pb = point_bbox(point);
            let mut result = None;
            for child in children.iter_mut() {
                if child.bbox().intersects(&pb) {
                    result = node_remove(child, id, point);
                    if result.is_some() {
                        break;
                    }
                }
            }
            // Prune empty children
            children.retain(|c| match c.as_ref() {
                RNode::Leaf { entries, .. } => !entries.is_empty(),
                RNode::Internal { children, .. } => !children.is_empty(),
            });
            if !children.is_empty() {
                *bbox = recompute_internal_bbox(children);
            }
            result
        }
    }
}

fn node_range<'a, T, C: CoordType, const D: usize>(
    node: &'a RNode<T, C, D>,
    bbox: &BBox<C, D>,
    out: &mut Vec<(EntryId, &'a T)>,
) {
    match node {
        RNode::Leaf { bbox: nb, entries } => {
            if !nb.intersects(bbox) {
                return;
            }
            for e in entries {
                if bbox.contains_point(&e.point) {
                    out.push((e.id, &e.payload));
                }
            }
        }
        RNode::Internal { bbox: nb, children } => {
            if !nb.intersects(bbox) {
                return;
            }
            for child in children {
                node_range(child, bbox, out);
            }
        }
    }
}

fn node_collect<'a, T, C: CoordType, const D: usize>(
    node: &'a RNode<T, C, D>,
    out: &mut Vec<(Point<C, D>, EntryId, &'a T)>,
) {
    match node {
        RNode::Leaf { entries, .. } => {
            for e in entries {
                out.push((e.point, e.id, &e.payload));
            }
        }
        RNode::Internal { children, .. } => {
            for child in children {
                node_collect(child, out);
            }
        }
    }
}

/// R*-tree spatial index.
///
/// Stores points with associated payloads and supports range queries, kNN
/// queries, and spatial joins. Uses forced reinsertion on overflow and
/// quadratic splits.
pub struct RTree<T, C, const D: usize> {
    root: Box<RNode<T, C, D>>,
    len: usize,
    next_id: u64,
    /// Map from EntryId → point, used for O(1) point lookup during remove.
    id_to_point: HashMap<u64, Point<C, D>>,
}

impl<T, C: CoordType, const D: usize> RTree<T, C, D> {
    /// Create an empty R-tree.
    pub fn new() -> Self {
        Self {
            root: Box::new(RNode::Leaf {
                bbox: BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D])),
                entries: Vec::new(),
            }),
            len: 0,
            next_id: 0,
            id_to_point: HashMap::new(),
        }
    }

    fn alloc_id(&mut self) -> EntryId {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn insert_entry(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        let id = self.alloc_id();
        self.id_to_point.insert(id.0, point);
        let entry = Entry { point, payload, id };

        let split = node_insert(&mut self.root, entry);
        if let Some(sibling) = split {
            let old_bbox = *self.root.bbox();
            let sib_bbox = *sibling.bbox();
            let new_bbox = merge_bbox(&old_bbox, &sib_bbox);
            let old_root = std::mem::replace(
                &mut self.root,
                Box::new(RNode::Internal {
                    bbox: new_bbox,
                    children: Vec::new(),
                }),
            );
            if let RNode::Internal { children, bbox } = &mut *self.root {
                children.push(old_root);
                children.push(sibling);
                *bbox = new_bbox;
            }
        }
        self.len += 1;
        id
    }

    pub fn remove_entry(&mut self, id: EntryId) -> Option<T> {
        let point = self.id_to_point.remove(&id.0)?;
        let result = node_remove(&mut self.root, id, &point);
        if result.is_some() {
            self.len -= 1;
            loop {
                let collapse = matches!(&*self.root,
                    RNode::Internal { children, .. } if children.len() == 1);
                if !collapse {
                    break;
                }
                let old = std::mem::replace(
                    &mut self.root,
                    Box::new(RNode::Leaf {
                        bbox: BBox::new(Point::new([C::zero(); D]), Point::new([C::zero(); D])),
                        entries: Vec::new(),
                    }),
                );
                if let RNode::Internal { mut children, .. } = *old {
                    self.root = children.remove(0);
                }
            }
        }
        result
    }

    /// Return all entries whose points lie within `bbox`.
    pub fn range_query_impl<'a>(&'a self, bbox: &BBox<C, D>) -> Vec<(EntryId, &'a T)> {
        let mut out = Vec::new();
        node_range(&self.root, bbox, &mut out);
        out
    }

    pub fn knn_query_impl<'a>(
        &'a self,
        point: &Point<C, D>,
        k: usize,
    ) -> Vec<(f64, EntryId, &'a T)> {
        if k == 0 {
            return Vec::new();
        }
        let mut raw: Vec<(Point<C, D>, EntryId, &'a T)> = Vec::new();
        node_collect(&self.root, &mut raw);
        let mut all: Vec<(f64, EntryId, &'a T)> = raw
            .into_iter()
            .map(|(p, id, payload)| (point_dist_sq(&p, point).sqrt(), id, payload))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    fn collect_all(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        let mut out = Vec::new();
        node_collect(&self.root, &mut out);
        out
    }
}

impl<T, C: CoordType, const D: usize> Default for RTree<T, C, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static, C: CoordType, const D: usize> SpatialBackend<T, C, D>
    for RTree<T, C, D>
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
            let bbox_a = point_bbox(pa);
            for (pb, id_b, _) in &other_entries {
                if bbox_a.intersects(&point_bbox(pb)) {
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
        BackendKind::RTree
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
    }

    fn rand_pts_2d(n: usize, seed: u64) -> Vec<Point<f64, 2>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]))
            .collect()
    }

    fn rand_pts_3d(n: usize, seed: u64) -> Vec<Point<f64, 3>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                Point::new([
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                ])
            })
            .collect()
    }

    fn rand_pts_6d(n: usize, seed: u64) -> Vec<Point<f64, 6>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                Point::new([
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                    rng.next_f64() * 1000.0,
                ])
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
        let mut tree = RTree::<u32, f64, 2>::new();
        assert_eq!(tree.len(), 0);
        tree.insert(Point::new([0.5, 0.5]), 1);
        assert_eq!(tree.len(), 1);
        tree.insert(Point::new([1.5, 1.5]), 2);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn range_query_basic() {
        let mut tree = RTree::<u32, f64, 2>::new();
        let id1 = tree.insert(Point::new([0.5, 0.5]), 1);
        let id2 = tree.insert(Point::new([1.5, 1.5]), 2);
        let _id3 = tree.insert(Point::new([5.0, 5.0]), 3);
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
    fn range_query_vs_brute_force_2d_10k() {
        let pts = rand_pts_2d(10_000, 42);
        let mut tree = RTree::<usize, f64, 2>::new();
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
        assert_eq!(got, expected, "2D 10k range query mismatch");
    }

    #[test]
    fn range_query_vs_brute_force_3d_10k() {
        let pts = rand_pts_3d(10_000, 43);
        let mut tree = RTree::<usize, f64, 3>::new();
        let mut pt_ids = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = tree.insert(p, i);
            pt_ids.push((p, id));
        }
        let bbox = BBox::new(
            Point::new([0.0, 0.0, 0.0]),
            Point::new([500.0, 500.0, 500.0]),
        );
        let mut got: Vec<EntryId> = tree
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        let expected = brute_range(&pt_ids, &bbox);
        assert_eq!(got, expected, "3D 10k range query mismatch");
    }

    #[test]
    fn range_query_vs_brute_force_6d_10k() {
        let pts = rand_pts_6d(10_000, 44);
        let mut tree = RTree::<usize, f64, 6>::new();
        let mut pt_ids = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = tree.insert(p, i);
            pt_ids.push((p, id));
        }
        let bbox = BBox::new(Point::new([0.0; 6]), Point::new([500.0; 6]));
        let mut got: Vec<EntryId> = tree
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        let expected = brute_range(&pt_ids, &bbox);
        assert_eq!(got, expected, "6D 10k range query mismatch");
    }

    #[test]
    fn knn_correctness_2d() {
        let pts = rand_pts_2d(1000, 99);
        let mut tree = RTree::<usize, f64, 2>::new();
        let mut all: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            let id = tree.insert(p, i);
            all.push((p, id));
        }
        let query = Point::new([500.0, 500.0]);
        let k = 5;
        let knn = tree.knn_query(&query, k);
        assert_eq!(knn.len(), k);
        for w in knn.windows(2) {
            assert!(w[0].0 <= w[1].0, "kNN not sorted");
        }
        let mut bf: Vec<(f64, EntryId)> = all
            .iter()
            .map(|(p, id)| (point_dist_sq(p, &query).sqrt(), *id))
            .collect();
        bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let bf_ids: Vec<EntryId> = bf[..k].iter().map(|(_, id)| *id).collect();
        let knn_ids: Vec<EntryId> = knn.iter().map(|(_, id, _)| *id).collect();
        assert_eq!(knn_ids, bf_ids, "kNN ids don't match brute force");
    }

    #[test]
    fn remove_works() {
        let mut tree = RTree::<&str, f64, 2>::new();
        let id1 = tree.insert(Point::new([1.0, 1.0]), "a");
        let id2 = tree.insert(Point::new([2.0, 2.0]), "b");
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.remove(id1), Some("a"));
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
    fn kind_is_rtree() {
        assert_eq!(RTree::<u32, f64, 2>::new().kind(), BackendKind::RTree);
    }

    #[test]
    fn bulk_load_correctness() {
        let pts = rand_pts_2d(500, 77);
        let entries: Vec<(Point<f64, 2>, usize)> =
            pts.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let tree = RTree::<usize, f64, 2>::bulk_load(entries);
        assert_eq!(tree.len(), 500);
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
        assert_eq!(tree.range_query(&bbox).len(), 500);
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

    fn pt3d() -> impl Strategy<Value = Point<f64, 3>> {
        (0.0_f64..1000.0, 0.0_f64..1000.0, 0.0_f64..1000.0)
            .prop_map(|(x, y, z)| Point::new([x, y, z]))
    }

    fn bbox3d() -> impl Strategy<Value = BBox<f64, 3>> {
        (
            0.0_f64..800.0,
            0.0_f64..800.0,
            0.0_f64..800.0,
            10.0_f64..200.0,
            10.0_f64..200.0,
            10.0_f64..200.0,
        )
            .prop_map(|(x, y, z, w, h, d)| {
                BBox::new(Point::new([x, y, z]), Point::new([x + w, y + h, z + d]))
            })
    }

    // For any dataset and bbox, range_query result must equal brute-force linear scan.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_range_query_oracle_2d(
            pts in prop::collection::vec(pt2d(), 1..100),
            bbox in bbox2d(),
        ) {
            let mut tree = RTree::<usize, f64, 2>::new();
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

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_range_query_oracle_3d(
            pts in prop::collection::vec(pt3d(), 1..80),
            bbox in bbox3d(),
        ) {
            let mut tree = RTree::<usize, f64, 3>::new();
            let mut pt_ids: Vec<(Point<f64, 3>, EntryId)> = Vec::new();
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

    // knn_query(k) must return globally nearest k points in ascending distance order.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_knn_correctness(
            pts in prop::collection::vec(pt2d(), 5..80),
            qx in 0.0_f64..1000.0,
            qy in 0.0_f64..1000.0,
            k in 1usize..5,
        ) {
            let mut tree = RTree::<usize, f64, 2>::new();
            let mut all: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                let id = tree.insert(*p, i);
                all.push((*p, id));
            }
            let query = Point::new([qx, qy]);
            let actual_k = k.min(all.len());
            let knn = tree.knn_query(&query, actual_k);
            prop_assert_eq!(knn.len(), actual_k);
            for w in knn.windows(2) {
                prop_assert!(w[0].0 <= w[1].0, "kNN not sorted");
            }
            let mut bf: Vec<(f64, EntryId)> = all
                .iter()
                .map(|(p, id)| (point_dist_sq(p, &query).sqrt(), *id))
                .collect();
            bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let bf_ids: Vec<EntryId> = bf[..actual_k].iter().map(|(_, id)| *id).collect();
            let knn_ids: Vec<EntryId> = knn.iter().map(|(_, id, _)| *id).collect();
            prop_assert_eq!(knn_ids, bf_ids);
        }
    }

    // spatial_join must return exactly the pairs where bboxes intersect.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_spatial_join_completeness(
            pts_a in prop::collection::vec(pt2d(), 1..30),
            pts_b in prop::collection::vec(pt2d(), 1..30),
        ) {
            let mut tree_a = RTree::<usize, f64, 2>::new();
            let mut tree_b = RTree::<usize, f64, 2>::new();
            let mut ids_a: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            let mut ids_b: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts_a.iter().enumerate() {
                let id = tree_a.insert(*p, i);
                ids_a.push((*p, id));
            }
            for (i, p) in pts_b.iter().enumerate() {
                let id = tree_b.insert(*p, i);
                ids_b.push((*p, id));
            }
            let mut got = tree_a.spatial_join(&tree_b);
            got.sort();
            let mut expected: Vec<(EntryId, EntryId)> = Vec::new();
            for (pa, id_a) in &ids_a {
                for (pb, id_b) in &ids_b {
                    if point_bbox(pa).intersects(&point_bbox(pb)) {
                        expected.push((*id_a, *id_b));
                    }
                }
            }
            expected.sort();
            prop_assert_eq!(got, expected);
        }
    }
}
