//! Grid spatial index backend.
//!
//! Implements the [`SpatialBackend`] trait using a uniform spatial hash grid
//! with D-dimensional cell coordinates addressed by `[i32; D]`.

use std::collections::HashMap;

use crate::backends::SpatialBackend;
use crate::types::{BBox, BackendKind, CoordType, EntryId, Point};

/// D-dimensional integer cell coordinate used as a `HashMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCoord<const D: usize>([i32; D]);

/// Uniform spatial hash grid index.
///
/// Points are bucketed into axis-aligned cells of size `cell_size[d]` along
/// each dimension. The cell coordinate for a point `p` along axis `d` is
/// `floor((p[d] - origin[d]) / cell_size[d])`.
///
/// # Type Parameters
/// - `T`: payload type
/// - `C`: coordinate scalar type (must implement [`CoordType`])
/// - `D`: number of spatial dimensions
pub struct GridIndex<T, C, const D: usize> {
    cells: HashMap<GridCoord<D>, Vec<(Point<C, D>, T, EntryId)>>,
    cell_size: [C; D],
    origin: Point<C, D>,
    len: usize,
    next_id: u64,
    id_to_cell: HashMap<u64, GridCoord<D>>,
}

fn cell_coord<C: CoordType, const D: usize>(
    point: &Point<C, D>,
    origin: &Point<C, D>,
    cell_size: &[C; D],
) -> GridCoord<D> {
    let mut coord = [0i32; D];
    for d in 0..D {
        let p: f64 = point.coords()[d].into();
        let o: f64 = origin.coords()[d].into();
        let s: f64 = cell_size[d].into();
        coord[d] = if s == 0.0 {
            0
        } else {
            ((p - o) / s).floor() as i32
        };
    }
    GridCoord(coord)
}

fn cell_range<C: CoordType, const D: usize>(
    bbox: &BBox<C, D>,
    origin: &Point<C, D>,
    cell_size: &[C; D],
) -> ([i32; D], [i32; D]) {
    let mut min_coord = [0i32; D];
    let mut max_coord = [0i32; D];
    for d in 0..D {
        let lo: f64 = bbox.min.coords()[d].into();
        let hi: f64 = bbox.max.coords()[d].into();
        let o: f64 = origin.coords()[d].into();
        let s: f64 = cell_size[d].into();
        if s == 0.0 {
            min_coord[d] = 0;
            max_coord[d] = 0;
        } else {
            min_coord[d] = ((lo - o) / s).floor() as i32;
            max_coord[d] = ((hi - o) / s).floor() as i32;
        }
    }
    (min_coord, max_coord)
}

/// Iterate over every `[i32; D]` coordinate in `[min, max]` (inclusive) and
/// call `f` for each. The last axis varies fastest.
fn for_each_cell_in_range<const D: usize, F: FnMut(GridCoord<D>)>(
    min: &[i32; D],
    max: &[i32; D],
    f: &mut F,
) {
    let mut current = *min;
    loop {
        f(GridCoord(current));

        let mut carry = true;
        for d in (0..D).rev() {
            if carry {
                if current[d] < max[d] {
                    current[d] += 1;
                    carry = false;
                } else {
                    current[d] = min[d];
                }
            }
        }
        if carry {
            break;
        }
    }
}

impl<T, C: CoordType, const D: usize> GridIndex<T, C, D> {
    /// Create an empty grid with the given cell size and origin.
    pub fn new(cell_size: [C; D], origin: Point<C, D>) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size,
            origin,
            len: 0,
            next_id: 0,
            id_to_cell: HashMap::new(),
        }
    }

    fn alloc_id(&mut self) -> EntryId {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Choose a cell size targeting approximately one point per cell for `n`
    /// uniformly distributed points over `bbox`.
    ///
    /// Formula: `cell_size[d] = bbox_span[d] / n^(1/D)`
    fn default_cell_size(bbox: &BBox<C, D>, n: usize) -> [C; D] {
        let n_f = (n.max(1) as f64).powf(1.0 / D as f64);
        let mut cs = [C::zero(); D];
        for (d, c) in cs.iter_mut().enumerate().take(D) {
            let span: f64 = (bbox.max.coords()[d] - bbox.min.coords()[d]).into();
            let s = (span / n_f).max(1.0);
            *c = C::from(s as f32);
        }
        cs
    }

    /// Return the number of occupied cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn insert_entry(&mut self, point: Point<C, D>, payload: T) -> EntryId {
        let id = self.alloc_id();
        let coord = cell_coord(&point, &self.origin, &self.cell_size);
        self.id_to_cell.insert(id.0, coord);
        self.cells
            .entry(coord)
            .or_default()
            .push((point, payload, id));
        self.len += 1;
        id
    }

    pub fn remove_entry(&mut self, id: EntryId) -> Option<T> {
        let coord = self.id_to_cell.remove(&id.0)?;
        let cell = self.cells.get_mut(&coord)?;
        let pos = cell.iter().position(|(_, _, eid)| *eid == id)?;
        let (_, payload, _) = cell.swap_remove(pos);
        if cell.is_empty() {
            self.cells.remove(&coord);
        }
        self.len -= 1;
        Some(payload)
    }

    pub fn range_query_impl<'a>(&'a self, bbox: &BBox<C, D>) -> Vec<(EntryId, &'a T)> {
        let (min_coord, max_coord) = cell_range(bbox, &self.origin, &self.cell_size);
        let mut out = Vec::new();
        for_each_cell_in_range(&min_coord, &max_coord, &mut |coord| {
            if let Some(cell) = self.cells.get(&coord) {
                for (point, payload, id) in cell {
                    if bbox.contains_point(point) {
                        out.push((*id, payload));
                    }
                }
            }
        });
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
        let mut all: Vec<(f64, EntryId, &'a T)> = self
            .cells
            .values()
            .flat_map(|cell| cell.iter())
            .map(|(p, payload, id)| (point_dist_sq(p, point).sqrt(), *id, payload))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    fn collect_all(&self) -> Vec<(Point<C, D>, EntryId, &T)> {
        self.cells
            .values()
            .flat_map(|cell| cell.iter())
            .map(|(p, payload, id)| (*p, *id, payload))
            .collect()
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

impl<T, C: CoordType, const D: usize> Default for GridIndex<T, C, D> {
    fn default() -> Self {
        Self::new([C::from(1.0_f32); D], Point::new([C::zero(); D]))
    }
}

impl<T: Send + Sync + 'static, C: CoordType, const D: usize> SpatialBackend<T, C, D>
    for GridIndex<T, C, D>
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
        if entries.is_empty() {
            return Self::default();
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
        let bbox = BBox::new(Point::new(min_c), Point::new(max_c));
        let cell_size = Self::default_cell_size(&bbox, entries.len());
        let origin = Point::new(min_c);
        let mut grid = Self::new(cell_size, origin);
        for (point, payload) in entries {
            grid.insert_entry(point, payload);
        }
        grid
    }

    fn len(&self) -> usize {
        self.len
    }

    fn kind(&self) -> BackendKind {
        BackendKind::Grid
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
        let mut grid = GridIndex::<u32, f64, 2>::default();
        assert_eq!(grid.len(), 0);
        grid.insert(Point::new([0.5, 0.5]), 1u32);
        assert_eq!(grid.len(), 1);
        grid.insert(Point::new([1.5, 1.5]), 2u32);
        assert_eq!(grid.len(), 2);
    }

    #[test]
    fn range_query_basic() {
        let mut grid = GridIndex::<u32, f64, 2>::default();
        let id1 = grid.insert(Point::new([0.5, 0.5]), 1u32);
        let id2 = grid.insert(Point::new([1.5, 1.5]), 2u32);
        let _id3 = grid.insert(Point::new([5.0, 5.0]), 3u32);
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([2.0, 2.0]));
        let mut got: Vec<EntryId> = grid
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        assert_eq!(got, vec![id1, id2]);
    }

    #[test]
    fn remove_works() {
        let mut grid = GridIndex::<u32, f64, 2>::default();
        let id1 = grid.insert(Point::new([1.0, 1.0]), 10u32);
        let id2 = grid.insert(Point::new([2.0, 2.0]), 20u32);
        assert_eq!(grid.len(), 2);
        assert_eq!(grid.remove(id1), Some(10u32));
        assert_eq!(grid.len(), 1);
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([3.0, 3.0]));
        let ids: Vec<EntryId> = grid
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(!ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn kind_is_grid() {
        assert_eq!(
            GridIndex::<u32, f64, 2>::default().kind(),
            BackendKind::Grid
        );
    }

    #[test]
    fn cell_coord_d2() {
        let origin = Point::new([0.0_f64, 0.0]);
        let cell_size = [1.0_f64, 1.0];
        let point = Point::new([2.5_f64, 3.7]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [2, 3]);
    }

    #[test]
    fn cell_coord_d3() {
        let origin = Point::new([0.0_f64; 3]);
        let cell_size = [2.0_f64; 3];
        let point = Point::new([5.0_f64, 7.0, 9.0]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [2, 3, 4]);
    }

    #[test]
    fn cell_coord_d4() {
        let origin = Point::new([0.0_f64; 4]);
        let cell_size = [10.0_f64; 4];
        let point = Point::new([15.0_f64, 25.0, 35.0, 45.0]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [1, 2, 3, 4]);
    }

    #[test]
    fn cell_coord_d5() {
        let origin = Point::new([0.0_f64; 5]);
        let cell_size = [5.0_f64; 5];
        let point = Point::new([0.0_f64, 5.0, 10.0, 15.0, 20.0]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn cell_coord_d6() {
        let origin = Point::new([0.0_f64; 6]);
        let cell_size = [3.0_f64; 6];
        let point = Point::new([3.0_f64, 6.0, 9.0, 12.0, 15.0, 18.0]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn cell_coord_negative() {
        let origin = Point::new([0.0_f64, 0.0]);
        let cell_size = [1.0_f64, 1.0];
        let point = Point::new([-1.5_f64, -0.5]);
        let coord = cell_coord(&point, &origin, &cell_size);
        assert_eq!(coord.0, [-2, -1]);
    }

    #[test]
    fn uniform_data_approx_one_point_per_cell() {
        let n = 10_000usize;
        let mut rng = Lcg::new(42);
        let entries: Vec<(Point<f64, 2>, usize)> = (0..n)
            .map(|i| {
                (
                    Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]),
                    i,
                )
            })
            .collect();
        let grid = GridIndex::<usize, f64, 2>::bulk_load(entries);
        let cell_count = grid.cells.len();
        let avg = n as f64 / cell_count as f64;
        assert!(
            (0.5..=4.0).contains(&avg),
            "avg points/cell = {avg:.2}, cell_count = {cell_count}"
        );
    }

    #[test]
    fn range_query_vs_brute_force_2d_10k() {
        let n = 10_000usize;
        let mut rng = Lcg::new(99);
        let mut grid = GridIndex::<usize, f64, 2>::new([10.0_f64, 10.0], Point::new([0.0, 0.0]));
        let mut pt_ids = Vec::new();
        for i in 0..n {
            let p = Point::new([rng.next_f64() * 1000.0, rng.next_f64() * 1000.0]);
            let id = grid.insert(p, i);
            pt_ids.push((p, id));
        }
        let bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([500.0, 500.0]));
        let mut got: Vec<EntryId> = grid
            .range_query(&bbox)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        got.sort_by_key(|id| id.0);
        let expected = brute_range(&pt_ids, &bbox);
        assert_eq!(got, expected, "2D 10k range query mismatch");
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

    // Insert-Remove Round Trip
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 100,
            ..Default::default()
        })]

        #[test]
        fn prop_insert_remove_round_trip_grid(
            pts in prop::collection::vec(pt2d(), 1..50),
            remove_indices in prop::collection::vec(0usize..50, 0..25),
        ) {
            let mut grid = GridIndex::<usize, f64, 2>::new(
                [10.0_f64, 10.0],
                Point::new([0.0_f64, 0.0]),
            );
            let mut inserted: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, &p) in pts.iter().enumerate() {
                let id = grid.insert(p, i);
                inserted.push((p, id));
            }
            let mut removed_ids: Vec<EntryId> = Vec::new();
            for &ri in &remove_indices {
                let idx = ri % inserted.len();
                let (_, id) = inserted[idx];
                if !removed_ids.contains(&id) {
                    let result = grid.remove(id);
                    prop_assert!(result.is_some(), "remove returned None for inserted id");
                    removed_ids.push(id);
                }
            }
            // Use a bbox that covers the point space (0..1000) without being huge
            let full_bbox = BBox::new(Point::new([0.0, 0.0]), Point::new([1000.0, 1000.0]));
            let remaining_ids: Vec<EntryId> = grid.range_query(&full_bbox)
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
            prop_assert_eq!(grid.len(), expected_len);
        }
    }

    // brute-force linear scan for any random dataset and bbox.
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 200,
            ..Default::default()
        })]

        #[test]
        fn prop_range_query_oracle_grid(
            pts in prop::collection::vec(pt2d(), 1..100),
            bbox in bbox2d(),
        ) {
            let mut grid = GridIndex::<usize, f64, 2>::new(
                [10.0_f64, 10.0],
                Point::new([0.0_f64, 0.0]),
            );
            let mut pt_ids: Vec<(Point<f64, 2>, EntryId)> = Vec::new();
            for (i, p) in pts.iter().enumerate() {
                let id = grid.insert(*p, i);
                pt_ids.push((*p, id));
            }
            let mut got: Vec<EntryId> =
                grid.range_query(&bbox).into_iter().map(|(id, _)| id).collect();
            got.sort_by_key(|id| id.0);
            let expected = brute_range(&pt_ids, &bbox);
            prop_assert_eq!(got, expected);
        }
    }
}
