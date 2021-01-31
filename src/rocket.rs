use crate::imgui_support;
use imgui::*;

use rapier2d::dynamics::{
    BodyStatus, IntegrationParameters, JointSet, RigidBodyBuilder, RigidBodyHandle, RigidBodySet, FixedJoint
};
use rapier2d::geometry::{
    BroadPhase, Collider, ColliderBuilder, ColliderHandle, ColliderSet, NarrowPhase,
};
use rapier2d::na::{Isometry2, Point2, Translation2, Vector2};
use rapier2d::pipeline::PhysicsPipeline;

pub fn main() {
    let system = imgui_support::init(file!());

    // Here the gravity is -9.81 along the y axis.
    let mut pipeline = PhysicsPipeline::new();
    let gravity = Vector2::new(0.0, -9.81);
    let integration_parameters = IntegrationParameters::default();
    let mut broad_phase = BroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut joints = JointSet::new();
    // We ignore contact events for now.
    let event_handler = ();

    let (b1, _) = add_cuboid(
        &mut bodies,
        &mut colliders,
        BodyStatus::Dynamic,
        Isometry2::new(Vector2::new(0.0, 20.0), 0.0),
        5.0,
        5.0,
    );
    // let (b2, _) = add_cuboid(
    //     &mut bodies,
    //     &mut colliders,
    //     BodyStatus::Dynamic,
    //     Isometry2::new(Vector2::new(3.0, 30.0), 0.0),
    //     5.0,
    //     5.0,
    // );

    // joints.insert(&mut bodies, b1, b2, FixedJoint::new(Isometry2::identity(), Isometry2::translation(-3.0, -10.0)));

    add_cuboid(
        &mut bodies,
        &mut colliders,
        BodyStatus::Static,
        Isometry2::new(Vector2::new(-7.5, 0.0), 0.0),
        5.0,
        1.0,
    );
    add_cuboid(
        &mut bodies,
        &mut colliders,
        BodyStatus::Static,
        Isometry2::new(Vector2::new(10.0, -5.0), 0.0),
        5.0,
        1.0,
    );

    system.main_loop(|_, ui| {
        pipeline.step(
            &gravity,
            &integration_parameters,
            &mut broad_phase,
            &mut narrow_phase,
            &mut bodies,
            &mut colliders,
            &mut joints,
            None,
            None,
            &event_handler,
        );

        Window::new(im_str!("Rocket"))
            .size([200.0, 150.0], Condition::FirstUseEver)
            .position([70.0, 70.0], Condition::FirstUseEver)
            .build(ui, || {
                let mouse = ui.io().mouse_pos;
                let pt = Point2::new(mouse[0] / 10.0 - 64.0, -mouse[1] / 10.0 + 38.0);
                if ui.io().mouse_down[0] {
                    bodies.get_mut(b1).unwrap().apply_force_at_point(Vector2::new(0.0, 2000.0), pt, true);
                }
                let dl = ui.get_background_draw_list();
                for (_, collider) in colliders.iter() {
                    draw_cuboid(&dl, collider, [0.2, 0.3, 0.6, 1.0]);
                }
            });
    });
}

fn add_cuboid(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    status: BodyStatus,
    position: Isometry2<f32>,
    hx: f32,
    hy: f32,
) -> (RigidBodyHandle, ColliderHandle) {
    let body = RigidBodyBuilder::new(status).position(position).build();
    let collider = ColliderBuilder::cuboid(hx, hy).build();
    let body_handle = bodies.insert(body);
    let collider_handle = colliders.insert(collider, body_handle, bodies);
    (body_handle, collider_handle)
}

fn point(pt: Point2<f32>) -> [f32; 2] {
    [pt[0] * 10.0, -pt[1] * 10.0]
}

fn draw_cuboid(dl: &WindowDrawList, collider: &Collider, color: [f32; 4]) {
    let translation = Translation2::from(Vector2::new(64.0, -38.40));
    let cube = collider.shape().as_cuboid().unwrap().half_extents;
    let isometry = collider.position();
    let transform = translation * isometry;
    let w = cube[0];
    let h = cube[1];
    let ul = transform * Point2::new(-w, h);
    let ur = transform * Point2::new(w, h);
    let bl = transform * Point2::new(-w, -h);
    let br = transform * Point2::new(w, -h);
    dl.add_triangle(point(ul), point(ur), point(bl), color)
        .filled(true)
        .build();
    dl.add_triangle(point(ur), point(bl), point(br), color)
        .filled(true)
        .build();
}
