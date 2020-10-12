use crossbeam::TrySendError;
use imgui::*;
use plotters::prelude::*;
use plotters_imgui::ImguiBackend;

mod easy_21;
mod imgui_support;
mod learn;

use easy_21::{HasV, HasQ};

fn main() {
    let mut system = imgui_support::init(file!());
    let style = system.imgui.style_mut().use_light_colors();
    style.window_rounding = 0.0;
    style.scrollbar_rounding = 0.0;

    let mut state = easy_21::ApproxState::init();

    let (req, resp) = learn::queryable_state(state.clone(), |rng, s| {
        // easy_21::monte_carlo_prediction(rng, easy_21::example_policy, s)
        // easy_21::monte_carlo_control(rng, s)
        // easy_21::td_lambda_prediction(rng, 0.5, easy_21::example_policy, s)
        // easy_21::td_lambda_control(rng, 0.6, s)
        easy_21::approx_td_lambda_control(rng, 0.1, s)
    });

    let mut pitch = 0.2;
    let mut yaw = 0.5;
    let mut rms: Vec<(f64, f64)> = vec![];

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
                    rms = vec![];
                    req.send(learn::Req::SetState {
                        state: easy_21::ApproxState::init(),
                    })
                    .unwrap();
                }

                ui.text(im_str!("Episodes: {}", state.episodes));
                ui.text(im_str!("RMS Error: {:>5.2}", state.rms_error));

                Slider::new(im_str!("Pitch"))
                    .range(0.0..=2.0)
                    .display_format(im_str!("%.2f"))
                    .build(ui, &mut pitch);

                Slider::new(im_str!("Yaw"))
                    .range(-2.0..=2.0)
                    .display_format(im_str!("%.2f"))
                    .build(ui, &mut yaw);

                let dl = ui.get_background_draw_list();
                let root = ImguiBackend::new(&ui, &dl, (1400, 700)).into_drawing_area();
                let areas = root.split_evenly((1, 2));

                // 3D
                let mut chart = ChartBuilder::on(&areas[0])
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
                                state.get_q(s, &easy_21::Action::Hit)
                                    >= state.get_q(s, &easy_21::Action::Stick)
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

                // 2D
                &mut rms.push((state.episodes as f64 / 1_000_000.0, state.rms_error));

                let last = rms.last().map_or(0.1, |x| x.0).max(1.0);

                let mut chart = ChartBuilder::on(&areas[1])
                    .margin(200)
                    .margin_left(50)
                    .margin_top(50)
                    .x_label_area_size(100)
                    .y_label_area_size(30)
                    .build_cartesian_2d(0.0..last, 0.0..10.0).unwrap();

                chart.configure_mesh().draw().unwrap();

                chart
                    .draw_series(LineSeries::new(
                        rms.clone(),
                        &BLACK,
                    )).unwrap();

            });
    });
}
