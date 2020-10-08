use crossbeam::TrySendError;
use imgui::*;
use plotters::prelude::*;
use plotters_imgui::ImguiBackend;

mod easy_21;
mod imgui_support;
mod learn;

fn main() {
    let mut system = imgui_support::init(file!());
    let style = system.imgui.style_mut().use_light_colors();
    style.window_rounding = 0.0;
    style.scrollbar_rounding = 0.0;

    let mut state = easy_21::TDState::init();

    // let (req, resp) = learn::queryable_state(state.clone(), easy_21::monte_carlo_control);

    let (req, resp) = learn::queryable_state(state.clone(), |rng, s| {
        easy_21::td_lambda_prediction(rng, 0.5, 0.001, easy_21::example_policy, s)
    });

    let mut pitch = 0.2;
    let mut yaw = 0.5;

    system.main_loop(|_, ui| {
        for s in resp.try_iter() {
            state = s
        }
        // Request state (to be available hopefully in the next frame).
        if let Err(TrySendError::Disconnected(_)) = req.try_send(()) {
            panic!("Could not request state: Disconnected")
        }

        Window::new(im_str!("Easy 21"))
            .size([200.0, 200.0], Condition::FirstUseEver)
            .position([700.0, 100.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Episodes: {}", state.episodes));

                Slider::new(im_str!("Pitch"))
                    .range(0.0..=2.0)
                    .display_format(im_str!("%.2f"))
                    .build(ui, &mut pitch);

                Slider::new(im_str!("Yaw"))
                    .range(-2.0..=2.0)
                    .display_format(im_str!("%.2f"))
                    .build(ui, &mut yaw);

                let dl = ui.get_background_draw_list();
                let root = ImguiBackend::new(&ui, &dl, (700, 700)).into_drawing_area();

                let mut chart = ChartBuilder::on(&root)
                    .build_cartesian_3d(1.0..21.0, -1.0..1.0, 1.0..10.0)
                    .unwrap();

                chart.with_projection(|mut p| {
                    p.yaw = -yaw;
                    p.pitch = pitch;
                    p.scale = 0.7;
                    p.into_matrix()
                });

                chart
                    .configure_axes()
                    .x_labels(10)
                    .y_labels(9)
                    .z_labels(10)
                    .draw()
                    .unwrap();

                chart
                    .draw_series(
                        (1..21)
                            .flat_map(|p| (1..10).map(|d| (p, d)).collect::<Vec<_>>())
                            .map(|(player, dealer)| {
                                let coord = |p: i32, d: i32| -> (f64, f64, f64) {
                                    (
                                        p as f64,
                                        state.v.get(&easy_21::State {
                                            player: p,
                                            dealer: d,
                                        }) as f64,
                                        d as f64,
                                    )
                                };
                                PathElement::new(
                                    vec![
                                        coord(player, dealer),
                                        coord(player + 1, dealer),
                                        coord(player + 1, dealer + 1),
                                        coord(player, dealer + 1),
                                        coord(player, dealer),
                                    ],
                                    BLACK.mix(0.6).stroke_width(1),
                                )
                            }),
                    )
                    .unwrap();
            });
    });
}
