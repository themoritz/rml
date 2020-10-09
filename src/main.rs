use crossbeam::TrySendError;
use imgui::*;
use plotters::prelude::*;
use plotters_imgui::ImguiBackend;

mod easy_21;
mod imgui_support;
mod learn;

use easy_21::HasV;

fn main() {
    let mut system = imgui_support::init(file!());
    let style = system.imgui.style_mut().use_light_colors();
    style.window_rounding = 0.0;
    style.scrollbar_rounding = 0.0;

    let mut state = easy_21::TDControlState::init();

    let (req, resp) = learn::queryable_state(state.clone(), |rng, s| {
        // easy_21::monte_carlo_prediction(rng, easy_21::example_policy, s)
        // easy_21::monte_carlo_control(rng, s)
        // easy_21::td_lambda_prediction(rng, 0.5, easy_21::example_policy, s)
        easy_21::td_lambda_control(rng, 0.5, s)
    });

    let mut pitch = 0.2;
    let mut yaw = 0.5;

    system.main_loop(|_, ui| {
        for s in resp.try_iter() {
            state = s
        }
        // Request state (to be available hopefully in the next frame).
        if let Err(TrySendError::Disconnected(_)) = req.try_send(learn::Req::GetState) {
            panic!("Could not request state: Disconnected")
        }

        Window::new(im_str!("Easy 21"))
            .size([200.0, 150.0], Condition::FirstUseEver)
            .position([70.0, 70.0], Condition::FirstUseEver)
            .build(ui, || {
                if ui.small_button(im_str!("Reset")) {
                    req.send(learn::Req::SetState {
                        state: easy_21::TDControlState::init(),
                    })
                    .unwrap();
                }

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

                let states: Vec<_> = (1..21)
                    .flat_map(|player| (1..10).map(move |dealer| easy_21::State { player, dealer }))
                    .collect();

                chart
                    .draw_series(states.iter().map(|s| {
                        let coord = |p: i32, d: i32| -> (f64, f64, f64) {
                            (
                                p as f64,
                                state.get_v(&easy_21::State {
                                    player: p,
                                    dealer: d,
                                }),
                                d as f64,
                            )
                        };
                        PathElement::new(
                            vec![
                                coord(s.player, s.dealer),
                                coord(s.player + 1, s.dealer),
                                coord(s.player + 1, s.dealer + 1),
                                coord(s.player, s.dealer + 1),
                                coord(s.player, s.dealer),
                            ],
                            BLACK.mix(0.6).stroke_width(1),
                        )
                    }))
                    .unwrap();

                chart
                    .draw_series(
                        states
                            .iter()
                            .filter(|s| {
                                state.q.get(s, &easy_21::Action::Hit).0
                                    >= state.q.get(s, &easy_21::Action::Stick).0
                            })
                            .map(|s| {
                                Polygon::new(
                                    vec![
                                        (s.player as f64 + 0.1, -1.0, s.dealer as f64 + 0.1),
                                        (s.player as f64 + 0.9, -1.0, s.dealer as f64 + 0.1),
                                        (s.player as f64 + 0.9, -1.0, s.dealer as f64 + 0.9),
                                        (s.player as f64 + 0.1, -1.0, s.dealer as f64 + 0.9),
                                    ],
                                    &BLUE.mix(0.6),
                                )
                            }),
                    )
                    .unwrap();
            });
    });
}
