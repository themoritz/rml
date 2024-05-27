use crate::easy_21;

pub struct App {
    easy_21: easy_21::Easy21,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            easy_21: easy_21::Easy21::new(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.easy_21.show(ctx);

        ctx.request_repaint();
    }
}
