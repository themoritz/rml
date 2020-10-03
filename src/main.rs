use crossbeam::TrySendError;
use imgui::*;

mod imgui_support;
mod learn;

fn main() {
    let mut system = imgui_support::init(file!());
    let style = system.imgui.style_mut().use_light_colors();
    style.window_rounding = 0.0;
    style.scrollbar_rounding = 0.0;

    let mut state: i32 = 0;

    let (req, resp) = learn::queryable_state(state, |s| *s = (*s + 1) % 10000000);

    system.main_loop(|_, ui| {
        for s in resp.try_iter() {
            state = s
        }
        // Request state (to be available hopefully in the next frame).
        if let Err(TrySendError::Disconnected(_)) = req.try_send(()) {
            panic!("Could not request state: Disconnected")
        }

        Window::new(im_str!("Controls"))
            .size([400.0, 200.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Hello world!"));
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.0},{:.0})",
                    mouse_pos[0], mouse_pos[1]
                ));
                ui.plot_lines(
                    im_str!("Value Fn"),
                    &[
                        mouse_pos[0] / 10.0,
                        mouse_pos[1] / 10.0,
                        state as f32 / 100000.0,
                    ],
                )
                .scale_min(0.0)
                .scale_max(100.0)
                .graph_size([100.0, 50.0])
                .build();
                ui.get_window_draw_list()
                    .add_circle(mouse_pos, 50.0, [0.3, 0.5, 0.7, 0.6])
                    .thickness(5.0)
                    .num_segments(50)
                    .build();
            });
    });
}
