use crate::ad;
use crate::imgui_support;
use imgui::*;
use nom::{
    bytes::complete::{tag, take},
    number::complete::be_u32,
    IResult,
};

pub fn main() {
    let system = imgui_support::init(file!());

    println!("{:#?}", ad::example());
    std::process::exit(0);

    let bytes = std::fs::read("./resources/train-images-idx3-ubyte").unwrap();
    let (_, images) = Images::parse_idx(&bytes).unwrap();

    let bytes = std::fs::read("./resources/train-labels-idx1-ubyte").unwrap();
    let (_, labels) = Labels::parse_idx(&bytes).unwrap();

    let mut index: u32 = 0;

    system.main_loop(|_, ui| {
        Window::new(im_str!("MNIST"))
            .size([400.0, 100.0], Condition::FirstUseEver)
            .position([70.0, 170.0], Condition::FirstUseEver)
            .build(ui, || {
                Slider::new(im_str!("Image"))
                    .range(0..=(images.num_images as u32) - 1)
                    .display_format(im_str!("%i"))
                    .build(ui, &mut index);
                ui.text(im_str!("Label: {}", labels.label(index as usize)));
                let dl = ui.get_background_draw_list();
                images.draw(&dl, index as usize);
            });
    });
}

struct Labels<'a> {
    num_images: usize,
    v: &'a [u8],
}

impl<'a> Labels<'a> {
    fn parse_idx(i: &'a [u8]) -> IResult<&[u8], Labels<'a>> {
        let (i, _) = tag([0x00, 0x00, 0x08, 0x01])(i)?; // magic bytes
        let (i, num_images) = be_u32(i)?;
        let (i, v) = take(num_images)(i)?;
        Ok((
            i,
            Labels {
                num_images: num_images as usize,
                v,
            },
        ))
    }

    fn label(&self, i: usize) -> u8 {
        assert!(
            i < self.num_images,
            "Choose among at most {} images",
            self.num_images
        );
        self.v[i]
    }
}

#[derive(Debug)]
struct Images<'a> {
    num_images: usize,
    num_rows: usize,
    num_cols: usize,
    v: &'a [u8],
}

impl<'a> Images<'a> {
    fn parse_idx(i: &'a [u8]) -> IResult<&[u8], Images<'a>> {
        let (i, _) = tag([0x00, 0x00, 0x08, 0x03])(i)?; // magic bytes
        let (i, num_images) = be_u32(i)?;
        let (i, num_rows) = be_u32(i)?;
        let (i, num_cols) = be_u32(i)?;
        let (i, v) = take(num_images * num_rows * num_cols)(i)?;
        Ok((
            i,
            Images {
                num_images: num_images as usize,
                num_rows: num_rows as usize,
                num_cols: num_cols as usize,
                v,
            },
        ))
    }

    fn image(&'a self, i: usize) -> &'a [u8] {
        assert!(
            i < self.num_images,
            "Choose among at most {} images",
            self.num_images
        );
        let image_size = (self.num_rows * self.num_cols) as usize;
        let offset = i * image_size;
        &self.v[offset..(offset + image_size)]
    }

    fn draw(&self, draw_list: &WindowDrawList, i: usize) {
        let image = self.image(i);
        let dx = 40.0;
        let dy = 30.0;
        let scale = 3.0;
        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                let c = [
                    0.0,
                    0.0,
                    0.0,
                    image[row * self.num_rows + col] as f32 / 255.0,
                ];
                draw_list
                    .add_rect(
                        [dx + (col as f32) * scale, dy + (row as f32) * scale],
                        [
                            dx + (col as f32 + 1.0) * scale,
                            dy + (row as f32 + 1.0) * scale,
                        ],
                        c,
                    )
                    .filled(true)
                    .build();
            }
        }
    }
}
