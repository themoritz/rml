use crossbeam::TrySendError;
use imgui::*;

mod easy_21;
mod imgui_support;
mod learn;

fn main() {
    let mut system = imgui_support::init(file!());
    let style = system.imgui.style_mut().use_light_colors();
    style.window_rounding = 0.0;
    style.scrollbar_rounding = 0.0;

    let mut state = easy_21::MCState::init();

    let (req, resp) = learn::queryable_state(state.clone(), easy_21::monte_carlo_control);

    system.main_loop(|_, ui| {
        for s in resp.try_iter() {
            state = s
        }
        // Request state (to be available hopefully in the next frame).
        if let Err(TrySendError::Disconnected(_)) = req.try_send(()) {
            panic!("Could not request state: Disconnected")
        }

        Window::new(im_str!("Easy 21"))
            .size([400.0, 200.0], Condition::FirstUseEver)
            .position([200.0, 350.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Episodes: {}", state.episodes));

                let dl = ui.get_background_draw_list();
                for player in 1..21 {
                    for dealer in 1..10 {
                        let s = easy_21::State { player, dealer };
                        let value = state.v.get(&s).0;
                        let color = [0.0, 0.0, 0.0, (value + 1.0) / 2.0];
                        let x = player as f32 * 50.0;
                        let y = dealer as f32 * 25.0;

                        let xoff = -7.0;
                        let yoff = -3.0;
                        if state.q.get(&s, &easy_21::Action::Hit).0
                            > state.q.get(&s, &easy_21::Action::Stick).0
                        {
                            dl.add_rect(
                                [x + xoff, y + yoff],
                                [x + 50.0 + xoff, y + 25.0 + yoff],
                                [0.0, 0.8, 0.4, 1.0],
                            )
                            .filled(true)
                            .build();
                        }
                        dl.add_text([x, y], color, im_str!("{:>5.2}", value));
                    }
                }
            });
    });
}
