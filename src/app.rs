use crate::easy_21;

pub struct App {
    easy_21: easy_21::Easy21,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Disable feathering
        cc.egui_ctx.tessellation_options_mut(|o| o.feathering = false);
        Self {
            easy_21: easy_21::Easy21::new(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.easy_21.ui_content(ui);

            ctx.request_repaint();
        });
    }
}
