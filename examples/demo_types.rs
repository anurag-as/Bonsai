use bonsai::{BBox, Point};

fn main() {
    // Construct a 3D point
    let p = Point::<f64, 3>::new([1.5, 2.5, 3.5]);
    println!("Point: {:?}", p);

    // Construct a 3D bounding box
    let bbox = BBox::<f64, 3>::new(Point::new([0.0, 0.0, 0.0]), Point::new([5.0, 5.0, 5.0]));
    println!("BBox:  {:?}", bbox);

    // contains_point
    let inside = bbox.contains_point(&p);
    println!("bbox.contains_point({:?}) = {}", p.coords(), inside);

    let outside = Point::<f64, 3>::new([6.0, 2.5, 3.5]);
    let not_inside = bbox.contains_point(&outside);
    println!(
        "bbox.contains_point({:?}) = {}",
        outside.coords(),
        not_inside
    );

    // intersects
    let overlapping = BBox::<f64, 3>::new(Point::new([3.0, 3.0, 3.0]), Point::new([8.0, 8.0, 8.0]));
    let disjoint = BBox::<f64, 3>::new(Point::new([6.0, 6.0, 6.0]), Point::new([9.0, 9.0, 9.0]));

    println!(
        "bbox.intersects(overlapping) = {}",
        bbox.intersects(&overlapping)
    );
    println!(
        "bbox.intersects(disjoint)    = {}",
        bbox.intersects(&disjoint)
    );
}
