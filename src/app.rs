pub struct App {

}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello World!");
        });
    }
}
