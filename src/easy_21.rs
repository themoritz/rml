use egui::Grid;
use plotters::prelude::*;
use rand::{
    distributions::{uniform::Uniform, Distribution},
    Rng,
};

#[derive(Copy, Clone, Debug, PartialEq)]
enum Algorithm {
    MonteCarloPrediction,
    MonteCarloControl,
    TDLambdaPrediction,
    TDLambdaControl,
    ApproxTDLambdaControl,
}

impl Algorithm {
    fn initial_state(&self) -> Box<dyn Easy21State> {
        match self {
            Self::MonteCarloPrediction => Box::new(MCState::init(example_policy)),
            Self::MonteCarloControl => Box::new(MCControlState::init()),
            Self::TDLambdaPrediction => Box::new(TDState::init(example_policy)),
            Self::TDLambdaControl => Box::new(TDControlState::init()),
            Self::ApproxTDLambdaControl => Box::new(ApproxState::init()),
        }

    }
}

pub struct Easy21 {
    rng: rand::prelude::ThreadRng,
    updates_per_frame: i32,
    algorithm: Algorithm,
    chart: egui_plotter::Chart<Box<dyn Easy21State>>,
    rms: Vec<(f64, f64)>,
}

impl Default for Easy21 {
    fn default() -> Self {
        Self::new()
    }
}

impl Easy21 {
    pub fn new() -> Self {
        let algorithm = Algorithm::MonteCarloControl;
        let state = algorithm.initial_state();
        let chart = egui_plotter::Chart::new(state)
            .mouse(egui_plotter::MouseConfig::enabled())
            .pitch(0.2)
            .yaw(-0.5)
            .builder_cb(Box::new(|area, transform, state| {
                let mut chart = ChartBuilder::on(area)
                    .build_cartesian_3d(1.0..21.0, -1.0..1.0, 1.0..10.0)
                    .unwrap();

                chart.with_projection(|mut p| {
                    p.yaw = transform.yaw;
                    p.pitch = transform.pitch;
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
                    .flat_map(|player| (1..10).map(move |dealer| State { player, dealer }))
                    .collect();

                chart
                    .draw_series(states.iter().map(|s| {
                        let coord = |p: i32, d: i32| -> (f64, f64, f64) {
                            (
                                p as f64,
                                state.get_v(&State {
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
                            plotters::prelude::BLACK.mix(0.6).stroke_width(1),
                        )
                    }))
                    .unwrap();

                chart
                    .draw_series(
                        states
                            .iter()
                            .filter(|s| {
                                state.policy(s) == Action::Hit
                            })
                            .map(|s| {
                                Polygon::new(
                                    vec![
                                        (s.player as f64 + 0.1, -1.0, s.dealer as f64 + 0.1),
                                        (s.player as f64 + 0.9, -1.0, s.dealer as f64 + 0.1),
                                        (s.player as f64 + 0.9, -1.0, s.dealer as f64 + 0.9),
                                        (s.player as f64 + 0.1, -1.0, s.dealer as f64 + 0.9),
                                    ],
                                    BLUE.mix(0.6),
                                )
                            }),
                    )
                    .unwrap();
            }));

        Self {
            rng: rand::thread_rng(),
            updates_per_frame: 50,
            algorithm,
            chart,
            rms: vec![],
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        let state = self.chart.get_data_mut();
        let start_time = web_time::Instant::now();
        for _ in 0..self.updates_per_frame {
            state.update(&mut self.rng);
        }
        let elapsed = start_time.elapsed();
        let target_time_per_frame = 1_000_000.0 / 80.0;
        self.updates_per_frame = (self.updates_per_frame as f64 * target_time_per_frame / elapsed.as_micros() as f64).round() as i32;

        egui::Window::new("Easy21").show(ctx, |ui| {
            egui::ComboBox::from_label("Algorithm")
                .selected_text(format!("{:?}", self.algorithm))
                .show_ui(ui, |ui| {
                    ui.style_mut().wrap = Some(false);
                    ui.set_min_width(60.0);
                    let algos = [
                        Algorithm::MonteCarloControl,
                        Algorithm::MonteCarloPrediction,
                        Algorithm::TDLambdaPrediction,
                        Algorithm::TDLambdaControl,
                        Algorithm::ApproxTDLambdaControl,
                    ];
                    for algo in algos {
                        if ui.selectable_value(&mut self.algorithm, algo, format!("{:?}", &algo)).clicked() {
                            self.rms = vec![];
                            *state = self.algorithm.initial_state();
                        }
                    }
                });

            ui.add_space(5.0);

            Grid::new("grid").num_columns(2).show(ui, |ui| {
                ui.label("Episodes:");
                ui.label(state.episodes().to_string());
                ui.end_row();

                ui.label("Updates per frame:");
                ui.label(self.updates_per_frame.to_string());
                ui.end_row();
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui(ui);
        });
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let space = ui.available_rect_before_wrap();
        let (left_rect, right_rect) = space.split_left_right_at_fraction(0.5);
        let left_ui = ui.child_ui(left_rect, egui::Layout::default());
        let right_ui = ui.child_ui(right_rect, egui::Layout::default());

        // 3D
        self.chart.draw(&left_ui);

        // 2D
        let area = egui_plotter::EguiBackend::new(&right_ui).into_drawing_area();
        let state = self.chart.get_data();
        if state.episodes() > 1 && ui.ctx().frame_nr() % 2 == 0 {
            self.rms.push((state.episodes() as f64 / 1_000_000.0, state.rms_error()));
        }

        let last = self.rms.last().map_or(0.1, |x| x.0).max(1.0);

        let mut chart = ChartBuilder::on(&area)
            .margin(200)
            .margin_left(50)
            .margin_top(50)
            .x_label_area_size(100)
            .y_label_area_size(30)
            .build_cartesian_2d(0.0..last, 0.0..10.0)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(self.rms.clone(), &BLACK))
            .unwrap();
    }
}

trait Easy21State: HasV {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng);
    fn episodes(&self) -> i32;
    fn policy(&self, state: &State) -> Action;
    fn rms_error(&self) -> f64;
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct State {
    pub dealer: i32,
    pub player: i32,
}

impl State {
    pub fn init<R: Rng>(rng: &mut R) -> Self {
        Self {
            dealer: Card::draw(rng).value,
            player: Card::draw(rng).value,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Action {
    /// Draw another card from the deck. Then decide again.
    Hit,
    /// No further cards. Dealer will take turns.
    Stick,
}

enum CardColor {
    Black,
    Red,
}

struct Card {
    value: i32,
    color: CardColor,
}

impl Card {
    fn add_to(&self, x: i32) -> i32 {
        x + self.value
            * match self.color {
                CardColor::Black => 1,
                CardColor::Red => -1,
            }
    }

    fn draw<R: Rng>(rng: &mut R) -> Self {
        let v = Uniform::new_inclusive(1, 10);
        Self {
            value: v.sample(rng),
            color: if rng.gen::<f64>() < (1.0 / 3.0) {
                CardColor::Red
            } else {
                CardColor::Black
            },
        }
    }
}

pub struct Sample {
    state: State,
    reward: i32,
    terminal: bool,
}

fn is_bust(x: i32) -> bool {
    !(1..=21).contains(&x)
}

fn signum(x: i32) -> i32 {
    match x.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

pub fn step<R: Rng>(rng: &mut R, state: State, action: Action) -> Sample {
    match action {
        Action::Hit => {
            let card = Card::draw(rng);
            let player = card.add_to(state.player);
            Sample {
                state: State { player, ..state },
                reward: if is_bust(player) { -1 } else { 0 },
                terminal: is_bust(player),
            }
        }
        Action::Stick => {
            let mut dealer = state.dealer;
            loop {
                if dealer >= 17 {
                    // Dealer sticks
                    break Sample {
                        state: State { dealer, ..state },
                        reward: signum(state.player - dealer),
                        terminal: true,
                    };
                } else {
                    // Dealer hits
                    let card = Card::draw(rng);
                    dealer = card.add_to(dealer);
                    if is_bust(dealer) {
                        break Sample {
                            state: State { dealer, ..state },
                            reward: 1,
                            terminal: true,
                        };
                    }
                }
            }
        }
    }
}

#[inline]
fn cube_index(point: [i32; 3], max: [i32; 3]) -> usize {
    let layers = point[2] * max[0] * max[1];
    let rows = point[1] * max[0];
    let bits = point[0];
    (layers + rows + bits) as usize
}

fn cube_size(max: [i32; 3]) -> usize {
    (max[0] * max[1] * max[2]) as usize
}

pub trait HasV {
    fn get_v(&self, state: &State) -> f64;
}

#[derive(Clone)]
pub struct V<T>(Vec<T>);

impl<T> V<T> {
    const MAX: [i32; 3] = [41, 41, 1];

    #[inline]
    fn index(state: &State) -> usize {
        let point = [state.player + 10, state.dealer + 10, 0];
        cube_index(point, Self::MAX)
    }

    pub fn init(v: T) -> Self
    where
        T: Clone,
    {
        V(vec![v; cube_size(Self::MAX)])
    }

    pub fn get(&self, state: &State) -> T
    where
        T: Copy,
    {
        self.0[Self::index(state)]
    }

    pub fn set(&mut self, state: &State, v: T) -> &mut Self {
        self.0[Self::index(state)] = v;
        self
    }

    fn update<F>(&mut self, state: &State, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        let i = Self::index(state);
        self.0[i] = f(&self.0[i]);
        self
    }

    fn map<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        for v in self.0.iter_mut() {
            *v = f(v);
        }
        self
    }

    fn zip_with<F, U>(&mut self, other: &V<U>, f: F) -> &mut Self
    where
        F: Fn(&T, &U) -> T,
    {
        self.0 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(t, u)| f(t, u))
            .collect();
        self
    }
}

#[derive(Clone)]
pub struct Q<T>(Vec<T>);

impl<T> Q<T> {
    const MAX: [i32; 3] = [41, 41, 2];

    #[inline]
    fn index(state: &State, action: &Action) -> usize {
        let point = [
            state.player + 10,
            state.dealer + 10,
            match action {
                Action::Hit => 0,
                Action::Stick => 1,
            },
        ];
        cube_index(point, Self::MAX)
    }

    pub fn init(v: T) -> Self
    where
        T: Clone,
    {
        Q(vec![v; cube_size(Self::MAX)])
    }

    pub fn get(&self, state: &State, action: &Action) -> T
    where
        T: Copy,
    {
        self.0[Self::index(state, action)]
    }

    pub fn set(&mut self, state: &State, action: &Action, v: T) -> &mut Self {
        self.0[Self::index(state, action)] = v;
        self
    }

    fn update<F>(&mut self, state: &State, action: &Action, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        let i = Self::index(state, action);
        self.0[i] = f(&self.0[i]);
        self
    }

    fn map<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        for v in self.0.iter_mut() {
            *v = f(v);
        }
        self
    }

    fn zip_with<F, U>(&mut self, other: &Q<U>, f: F) -> &mut Self
    where
        F: Fn(&T, &U) -> T,
    {
        self.0 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(t, u)| f(t, u))
            .collect();
        self
    }
}

pub trait HasQ {
    fn get_q(&self, state: &State, action: &Action) -> f64;
}

impl HasQ for Q<(f64, i32)> {
    fn get_q(&self, state: &State, action: &Action) -> f64 {
        self.get(state, action).0
    }
}

pub fn example_policy<R: Rng>(_rng: &mut R, state: &State) -> Action {
    if state.player >= 20 {
        Action::Stick
    } else {
        Action::Hit
    }
}

pub fn episode<R: Rng, P: Fn(&mut R, &State) -> Action>(
    rng: &mut R,
    policy: P,
) -> (Vec<(State, Action)>, i32) {
    let mut state = State::init(rng);
    let mut state_actions = vec![];
    let reward = loop {
        let action = policy(rng, &state);
        state_actions.push((state, action));
        let sample = step(rng, state, action);
        state = sample.state;
        if sample.terminal {
            break sample.reward;
        }
    };
    (state_actions, reward)
}

#[derive(Clone)]
pub struct MCState {
    pub v: V<(f64, i32)>,
    pub episodes: i32,
    pub policy: fn(&mut rand::prelude::ThreadRng, &State) -> Action,
}

impl MCState {
    pub fn init(policy: fn(&mut rand::prelude::ThreadRng, &State) -> Action) -> Self {
        Self {
            v: V::init((0.0, 0)),
            episodes: 0,
            policy,
        }
    }
}

impl HasV for MCState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
    }
}

impl Easy21State for MCState {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng) {
        monte_carlo_prediction(rng, self.policy, self);
    }
    fn episodes(&self) -> i32 {
        self.episodes
    }
    fn policy(&self, state: &State) -> Action {
        (self.policy)(&mut rand::thread_rng(), state)
    }
    fn rms_error(&self) -> f64 {
        0.0
    }
}

pub fn monte_carlo_prediction<R: Rng, P: Fn(&mut R, &State) -> Action>(
    rng: &mut R,
    policy: P,
    mc_state: &mut MCState,
) {
    mc_state.episodes += 1;
    let (state_actions, reward) = episode(rng, policy);
    for (state, _) in state_actions {
        let (value, n) = mc_state.v.get(&state);
        let new_n = n + 1;
        let new_value = value + 1.0 / (new_n as f64) * (reward as f64 - value);
        mc_state.v.set(&state, (new_value, new_n));
    }
}

fn epsilon_greedy<R: Rng, Q: HasQ>(rng: &mut R, eps: f64, q: &Q, state: &State) -> Action {
    let q_hit = q.get_q(state, &Action::Hit);
    let q_stick = q.get_q(state, &Action::Stick);
    let threshold = if q_hit > q_stick {
        eps / 2.0
    } else {
        1.0 - eps / 2.0
    };
    if rng.gen::<f64>() < threshold {
        Action::Stick
    } else {
        Action::Hit
    }
}

fn greedy_episode<R: Rng>(rng: &mut R, mc_state: &MCControlState) -> (Vec<(State, Action)>, i32) {
    episode(rng, |rng, state| {
        let visited = mc_state.v.get(state).1 as f64;
        let eps = 1.0 / (10.0 + visited / 100_000.0);
        epsilon_greedy(rng, eps, &mc_state.q, state)
    })
}

#[derive(Clone)]
pub struct MCControlState {
    pub v: V<(f64, i32)>,
    pub q: Q<(f64, i32)>,
    pub episodes: i32,
}

impl MCControlState {
    pub fn init() -> Self {
        Self {
            v: V::init((0.0, 0)),
            q: Q::init((0.0, 0)),
            episodes: 0,
        }
    }
}

impl HasV for MCControlState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
    }
}

impl Easy21State for MCControlState {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng) {
        monte_carlo_control(rng, self);
    }
    fn episodes(&self) -> i32 {
        self.episodes
    }
    fn policy(&self, state: &State) -> Action {
        if self.q.get(state, &Action::Hit) > self.q.get(state, &Action::Stick) {
            Action::Hit
        } else {
            Action::Stick
        }
    }
    fn rms_error(&self) -> f64 {
        self.q.rms_error()
    }
}

pub fn monte_carlo_control<R: Rng>(rng: &mut R, mc_state: &mut MCControlState) {
    let (state_actions, reward) = greedy_episode(rng, mc_state);
    for (state, action) in state_actions {
        // Update Q
        let (value, n) = mc_state.q.get(&state, &action);
        let new_n = n + 1;
        let new_value = value + 1.0 / (new_n as f64) * (reward as f64 - value);
        mc_state.q.set(&state, &action, (new_value, new_n));

        // Update V
        let q_hit = mc_state.q.get(&state, &Action::Hit).0;
        let q_stick = mc_state.q.get(&state, &Action::Stick).0;
        mc_state
            .v
            .update(&state, |(_, n)| (q_hit.max(q_stick), n + 1));
    }
    // Inc episodes
    mc_state.episodes += 1;
    if mc_state.episodes == 200_000_000 {
        println!(
            "{:?}",
            mc_state.q.0.iter().map(|(v, _)| v).collect::<Vec<_>>()
        );
    }
}

#[derive(Clone)]
pub struct TDState {
    pub v: V<(f64, i32)>,
    pub eligibility_traces: V<f64>,
    pub episodes: i32,
    pub policy: fn(&mut rand::prelude::ThreadRng, &State) -> Action,
}

impl TDState {
    pub fn init(policy: fn(&mut rand::prelude::ThreadRng, &State) -> Action) -> Self {
        Self {
            v: V::init((0.0, 0)),
            eligibility_traces: V::init(0.0),
            episodes: 0,
            policy,
        }
    }
}

impl HasV for TDState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
    }
}

impl Easy21State for TDState {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng) {
        td_lambda_prediction(rng, 0.5, self.policy, self);
    }
    fn episodes(&self) -> i32 {
        self.episodes
    }
    fn policy(&self, state: &State) -> Action {
        (self.policy)(&mut rand::thread_rng(), state)
    }
    fn rms_error(&self) -> f64 {
        0.0
    }
}

/// One episode, will be looped over by the main loop.
pub fn td_lambda_prediction<R: Rng, P: Fn(&mut R, &State) -> Action>(
    rng: &mut R,
    lambda: f64,
    policy: P,
    td_state: &mut TDState,
) {
    td_state.eligibility_traces.map(|_| 0.0);
    let mut state = State::init(rng);
    loop {
        let action = policy(rng, &state);
        let sample = step(rng, state, action);

        let next_state = sample.state;

        // Update eligibility traces
        td_state.eligibility_traces.map(|v| v * lambda);
        td_state.eligibility_traces.update(&state, |v| v + 1.0);

        let td_error =
            (sample.reward as f64) + td_state.v.get(&next_state).0 - td_state.v.get(&state).0;
        td_state
            .v
            .zip_with(&td_state.eligibility_traces, |(v, n), eligibility| {
                let alpha = 1.0 / (10.0 + *n as f64);
                (v + alpha * td_error * eligibility, *n)
            });
        td_state.v.update(&state, |(v, n)| (*v, n + 1));

        if sample.terminal {
            break;
        } else {
            state = next_state;
        }
    }
    td_state.episodes += 1;
}

#[derive(Clone)]
pub struct TDControlState {
    pub v: V<(f64, i32)>,
    pub q: Q<(f64, i32)>,
    pub eligibility_traces: Q<f64>,
    pub episodes: i32,
    pub rms_error: f64,
}

impl TDControlState {
    pub fn init() -> Self {
        Self {
            v: V::init((0.0, 0)),
            q: Q::init((0.0, 0)),
            eligibility_traces: Q::init(0.0),
            episodes: 0,
            rms_error: 0.0,
        }
    }
}

impl HasV for TDControlState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
    }
}

impl Easy21State for TDControlState {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng) {
        td_lambda_control(rng, 0.6, self);
    }
    fn episodes(&self) -> i32 {
        self.episodes
    }
    fn policy(&self, state: &State) -> Action {
        if self.q.get(state, &Action::Hit) > self.q.get(state, &Action::Stick) {
            Action::Hit
        } else {
            Action::Stick
        }
    }
    fn rms_error(&self) -> f64 {
        self.rms_error
    }
}

pub fn td_lambda_control<R: Rng>(rng: &mut R, lambda: f64, td_state: &mut TDControlState) {
    td_state.eligibility_traces.map(|_| 0.0);
    let mut state = State::init(rng);

    let eps = 1.0 / (10.0 + td_state.v.get(&state).1 as f64 / 10_000.0);
    let mut action = epsilon_greedy(rng, eps, &td_state.q, &state);

    loop {
        let sample = step(rng, state, action);
        let next_state = sample.state;

        let eps = 1.0 / (10.0 + td_state.v.get(&next_state).1 as f64 / 10_000.0);
        let next_action = epsilon_greedy(rng, eps, &td_state.q, &next_state);

        // Update eligibility traces
        td_state.eligibility_traces.map(|v| v * lambda);
        td_state
            .eligibility_traces
            .update(&state, &action, |v| v + 1.0);

        let td_error = (sample.reward as f64) + td_state.q.get(&next_state, &next_action).0
            - td_state.q.get(&state, &action).0;
        td_state
            .q
            .zip_with(&td_state.eligibility_traces, |(v, n), eligibility| {
                let alpha = 1.0 / (10.0 + *n as f64);
                (v + alpha * td_error * eligibility, *n)
            });
        td_state.q.update(&state, &action, |(v, n)| (*v, *n + 1));

        // Update V
        let q_hit = td_state.q.get(&state, &Action::Hit).0;
        let q_stick = td_state.q.get(&state, &Action::Stick).0;
        td_state
            .v
            .update(&state, |(_, n)| (q_hit.max(q_stick), n + 1));

        if sample.terminal {
            break;
        } else {
            state = next_state;
            action = next_action;
        }
    }
    td_state.episodes += 1;
    if td_state.episodes % 1000 == 0 {
        td_state.rms_error = td_state.q.rms_error();
    }
}

#[derive(Clone)]
pub struct Vector {
    w: Vec<f64>,
}

impl Vector {
    fn init() -> Self {
        let mut w = Vec::with_capacity(36);
        w.resize(36, 0.0);
        Self { w }
    }

    pub fn cuboid_features(state: &State, action: &Action) -> Self {
        let mut result = Vec::with_capacity(36);
        for dealer_interval in &[1..=4, 4..=7, 7..=10] {
            for player_interval in &[1..=6, 4..=9, 7..=12, 10..=15, 13..=18, 16..=21] {
                for a in &[Action::Hit, Action::Stick] {
                    let in_dealer = dealer_interval.contains(&state.dealer);
                    let in_player = player_interval.contains(&state.player);
                    result.push(if in_dealer && in_player && action == a {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
        }
        Self { w: result }
    }

    fn get_q(&self, state: &State, action: &Action) -> f64 {
        Self::cuboid_features(state, action)
            .w
            .iter()
            .zip(&self.w)
            .map(|(a, b)| a * b)
            .sum()
    }

    fn zip_with<F>(&mut self, other: &Vector, f: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        self.w = self
            .w
            .iter()
            .zip(&other.w)
            .map(|(a, b)| f(*a, *b))
            .collect();
    }
}

impl HasQ for Vector {
    fn get_q(&self, state: &State, action: &Action) -> f64 {
        self.get_q(state, action)
    }
}

#[derive(Clone)]
pub struct ApproxState {
    pub q: Vector,
    pub eligibility_traces: Vector,
    pub episodes: i32,
    pub rms_error: f64,
}

impl ApproxState {
    pub fn init() -> Self {
        Self {
            q: Vector::init(),
            eligibility_traces: Vector::init(),
            episodes: 0,
            rms_error: 0.0,
        }
    }
}

impl HasV for ApproxState {
    fn get_v(&self, state: &State) -> f64 {
        let q_hit = self.q.get_q(state, &Action::Hit);
        let q_stick = self.q.get_q(state, &Action::Stick);
        q_hit.max(q_stick)
    }
}

impl HasQ for ApproxState {
    fn get_q(&self, state: &State, action: &Action) -> f64 {
        self.q.get_q(state, action)
    }
}

impl Easy21State for ApproxState {
    fn update(&mut self, rng: &mut rand::prelude::ThreadRng) {
        approx_td_lambda_control(rng, 0.1, self);
    }
    fn episodes(&self) -> i32 {
        self.episodes
    }
    fn policy(&self, state: &State) -> Action {
        if self.get_q(state, &Action::Hit) > self.get_q(state, &Action::Stick) {
            Action::Hit
        } else {
            Action::Stick
        }
    }
    fn rms_error(&self) -> f64 {
        self.rms_error
    }
}

pub fn approx_td_lambda_control<R: Rng>(rng: &mut R, lambda: f64, approx_state: &mut ApproxState) {
    approx_state.eligibility_traces = Vector::init();
    let mut state = State::init(rng);

    let eps = 0.05;
    let mut action = epsilon_greedy(rng, eps, &approx_state.q, &state);

    loop {
        let sample = step(rng, state, action);
        let next_state = sample.state;

        let next_action = epsilon_greedy(rng, eps, &approx_state.q, &next_state);

        // Update eligibility traces
        approx_state
            .eligibility_traces
            .zip_with(&Vector::cuboid_features(&state, &action), |e, x| {
                lambda * e + x
            });

        let td_error = (sample.reward as f64) + approx_state.q.get_q(&next_state, &next_action)
            - approx_state.q.get_q(&state, &action);
        approx_state
            .q
            .zip_with(&approx_state.eligibility_traces, |w, eligibility| {
                let alpha = 0.0001;
                w + alpha * td_error * eligibility
            });

        if sample.terminal {
            break;
        } else {
            state = next_state;
            action = next_action;
        }
    }
    approx_state.episodes += 1;
    if approx_state.episodes % 1000 == 0 {
        approx_state.rms_error = 0.0 // approx_state.q.rms_error();
    }
}

impl Q<(f64, i32)> {
    fn rms_error(&self) -> f64 {
        let optimal = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.11491226520118117,
            -0.06786737907363498,
            -0.030619839104085844,
            0.02775868450272315,
            0.061262189381693796,
            0.10621665874399272,
            0.1555898930659694,
            0.2121437686091034,
            0.2802836928031776,
            0.35233777664983607,
            0.42925827494563323,
            0.34407244146623917,
            0.2398061694835128,
            0.18254739912493953,
            0.06708229426433908,
            -0.009306365554038989,
            -0.10963455149501682,
            -0.193317732709308,
            -0.297388716005953,
            -0.4092653871608203,
            -0.5230769230769236,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.15451339915373635,
            -0.1137133887481722,
            -0.0705105633802819,
            -0.025558656766198024,
            0.012102283818075186,
            0.0543795880550208,
            0.1076041187502205,
            0.1536284546312137,
            0.22808896066281567,
            0.30187863383590896,
            0.38025125355902867,
            0.2966944943840231,
            0.20942916010180543,
            0.12076496281430746,
            0.032867910539972706,
            -0.04432937610507716,
            -0.13295370141124052,
            -0.22555893636724395,
            -0.32519043866561637,
            -0.43216896831843926,
            -0.5516467065868265,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.19536358938721568,
            -0.14654337296345368,
            -0.10503668171557536,
            -0.06577327564981808,
            -0.029844694766160685,
            0.013401313576575107,
            0.05375605460149762,
            0.11313887863283897,
            0.18295655834020405,
            0.25710431103823017,
            0.3314964501583053,
            0.2528369643284729,
            0.1905268245529248,
            0.08493743984600566,
            0.014823761941363759,
            -0.049426301853486,
            -0.1464570858283433,
            -0.25029469548133604,
            -0.3437168610816545,
            -0.44753511429358345,
            -0.5609386828160483,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.21008193801211164,
            -0.17538001933046252,
            -0.14231569956760762,
            -0.10002115357495371,
            -0.06990163701573071,
            -0.021642909420415836,
            0.015474564996674099,
            0.07142732370255199,
            0.14453983168283382,
            0.21653032706435663,
            0.2938813980573196,
            0.21884597985762266,
            0.1464528213533124,
            0.0506744440393731,
            -0.01988047322844251,
            -0.06458797327394196,
            -0.1585381222432264,
            -0.25984354628422407,
            -0.3548343657419879,
            -0.4483735996760701,
            -0.5844961240310086,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2435895170268789,
            -0.20645002375213387,
            -0.1739972532309745,
            -0.1369076697606272,
            -0.09930191972076824,
            -0.05834504567814508,
            -0.017246511302155376,
            0.03611378555798728,
            0.10549587198435352,
            0.18136593115015046,
            0.25978829599981196,
            0.1838975950644522,
            0.1149926675945047,
            0.04564315352697091,
            -0.021732858928743163,
            -0.09004133044676231,
            -0.18575572671801516,
            -0.27386587771203197,
            -0.37329255861365934,
            -0.47527891955372975,
            -0.5744125326370759,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.27852384108270123,
            -0.22820869565217436,
            -0.20029916165165054,
            -0.16349691661916432,
            -0.12039571710731506,
            -0.09318406322028257,
            -0.04723023010927184,
            0.0019072140370952945,
            0.07787374166557515,
            0.15049777100223782,
            0.2322769894004378,
            0.1591973727771936,
            0.08931224901653322,
            0.022931830633169035,
            -0.0655301845480141,
            -0.11994868505452196,
            -0.18766108247422658,
            -0.28786023348844725,
            -0.3706597921533877,
            -0.48783783783783885,
            -0.6012031139419687,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.33104459587682467,
            -0.2841134384966806,
            -0.24320319768443688,
            -0.22101860449121533,
            -0.17552937972356536,
            -0.1597104283533206,
            -0.10790774299835315,
            -0.05206672495671124,
            0.019142135443477693,
            0.0961110218802743,
            0.1794773445607034,
            0.11318514725009655,
            0.046558623334840905,
            -0.016420994295421824,
            -0.09542743538767391,
            -0.1453172205438066,
            -0.22816421001340034,
            -0.32323987538940846,
            -0.4060579728136082,
            -0.5018618506795725,
            -0.6012630662020908,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.38128457604119714,
            -0.34740869685114084,
            -0.30124530876833583,
            -0.2748016042324442,
            -0.24459988712353048,
            -0.21574503793262761,
            -0.17731857235032436,
            -0.12170442501324903,
            -0.044085019132882305,
            0.03507587566000208,
            0.12112500147282204,
            0.05802993848060112,
            -0.0019452718850165546,
            -0.06202731559747433,
            -0.11872603490100896,
            -0.17585015457355768,
            -0.24410207029369296,
            -0.3300396462336063,
            -0.4195273631840809,
            -0.5112491000719918,
            -0.6316855753646693,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.4412216345333551,
            -0.39317547341159326,
            -0.3555092507015711,
            -0.32415521476465725,
            -0.29958776881943167,
            -0.2788138632584532,
            -0.2358623199192821,
            -0.18023687920556775,
            -0.1100223593361758,
            -0.029017707801376656,
            0.05740426948968794,
            0.00035883193030051847,
            -0.05629316779741276,
            -0.11026174001039385,
            -0.16158840742187114,
            -0.21309033901626462,
            -0.2780706585884008,
            -0.35184598320392957,
            -0.44065742267041713,
            -0.5268760678144307,
            -0.6365723029839313,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.4876375952582563,
            -0.4490466909583913,
            -0.41903708915468835,
            -0.38586253771426776,
            -0.3609585290164124,
            -0.3383834548400181,
            -0.29713587779121853,
            -0.247434215378843,
            -0.18026142931930386,
            -0.10015519728161736,
            -0.010927569586099766,
            -0.06498681033376622,
            -0.11433335009236398,
            -0.16548244144635044,
            -0.2113559410967657,
            -0.2591830025255648,
            -0.3114327592524731,
            -0.3858151854031789,
            -0.4597217873596612,
            -0.5684354361193283,
            -0.6447971781305095,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3168706937083459,
            0.31727155110871075,
            0.3171743578086117,
            0.3175906235606465,
            0.3171138960758811,
            0.31783995656555397,
            0.3174790858326245,
            0.31594987619235393,
            0.31709862805375544,
            0.32060013654605163,
            0.3233732559575261,
            0.3164049983979489,
            0.32208054306944117,
            0.3165370149741887,
            0.32078045308979136,
            0.31507517467594726,
            0.4063164174652234,
            0.5769814061962328,
            0.7222632752540836,
            0.8497623283851594,
            0.9535849810265585,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25662148322602873,
            0.25524284089066335,
            0.2557964717491214,
            0.255786602950473,
            0.25545180177758176,
            0.25485154366743284,
            0.25585445486222624,
            0.2559006702019785,
            0.25488714372870336,
            0.2509306717021849,
            0.25752343364578245,
            0.24248642124321151,
            0.2521696207181575,
            0.2551437695963584,
            0.2564587271581569,
            0.2547692943084328,
            0.35160616252039495,
            0.5328122435063714,
            0.6971528798048026,
            0.8329829015749606,
            0.9489739102768059,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.20044513271503547,
            0.20111479841889957,
            0.1991708483612874,
            0.20144575148794047,
            0.2010188488005519,
            0.20137733540680147,
            0.20186688204607922,
            0.19982519904093596,
            0.20145460358835873,
            0.2010686164229486,
            0.20215701028342117,
            0.21324717285945002,
            0.1999008993956,
            0.20211266369927114,
            0.2006775940638971,
            0.20160160401971455,
            0.30703995852312815,
            0.4936511247211132,
            0.6683448182503812,
            0.8154756011432122,
            0.9422406967537619,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.1510017796404597,
            0.1528746493357407,
            0.15147015656416285,
            0.15308361465508738,
            0.15221660890167574,
            0.14996200452038325,
            0.15196657803894836,
            0.15260705525880822,
            0.15206524611641387,
            0.15032656411720555,
            0.15016282716198395,
            0.15667936727860468,
            0.15147767031431406,
            0.1504201827693027,
            0.14973573252568334,
            0.15166264097488508,
            0.2602925848719479,
            0.4610191967668689,
            0.6427047084382823,
            0.7998067939709153,
            0.9353223948194271,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.10748442229990916,
            0.10741130673487607,
            0.10657116440721406,
            0.10783988215456847,
            0.10765978875241491,
            0.10641063686568766,
            0.10751215454213317,
            0.10765242570034726,
            0.1069262412525317,
            0.11110717197858636,
            0.1197799959258502,
            0.08450289527922275,
            0.09508389755666777,
            0.10776001437039734,
            0.10693205591588073,
            0.10564278976862908,
            0.2208740027110332,
            0.4271774208948383,
            0.6206821438252732,
            0.7875788149917154,
            0.9329004329004278,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0698649899955982,
            0.06886966805001381,
            0.07019714127735399,
            0.06783967377704865,
            0.07080871096615295,
            0.06981813545549646,
            0.06982330448476157,
            0.07055785767140832,
            0.06864684865321226,
            0.0717938820804786,
            0.05797450084195352,
            0.0746526414946229,
            0.06520588546628582,
            0.07070743067706722,
            0.06686082224647191,
            0.06765248998241143,
            0.18546989557876417,
            0.39835483689222234,
            0.5999581470560069,
            0.7784472407535921,
            0.9276614600642554,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0256333644923194,
            -0.026354378549853456,
            -0.026687781999803477,
            -0.026163663489201194,
            -0.026374458849310816,
            -0.02653535049884668,
            -0.02647140944926371,
            -0.025400634007980597,
            -0.027159119029943068,
            -0.02589552495105952,
            -0.03620985859653759,
            -0.03683702989392465,
            -0.027232580961727156,
            -0.01923302176306596,
            -0.026551641504912193,
            -0.029370306429570545,
            0.14770427172101724,
            0.4209621006401787,
            0.6115667037412217,
            0.7822005796495902,
            0.9301553909714897,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.11967703153876463,
            -0.11956321443635208,
            -0.11946264609437775,
            -0.11991061581298706,
            -0.1186146148267004,
            -0.12068032104538218,
            -0.11898122827346212,
            -0.11904083273663528,
            -0.12295852630845804,
            -0.12242428374209682,
            -0.11212121212121211,
            -0.1206382150732907,
            -0.11724680874992924,
            -0.10383584244174139,
            -0.11904769083722135,
            -0.12043948515947298,
            0.05136777061260892,
            0.3762041093089053,
            0.6242304864384036,
            0.7883013693963756,
            0.933783231083846,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.2096854114217828,
            -0.2100423515543988,
            -0.20924228618391802,
            -0.21108751817782284,
            -0.20928887424369574,
            -0.2098251628415254,
            -0.20983648651245793,
            -0.20268521514073087,
            -0.20699017007333353,
            -0.20977066260084862,
            -0.20101464109296496,
            -0.20890202555854945,
            -0.22059894606057945,
            -0.2165551839464881,
            -0.2176614881439092,
            -0.2083310310858574,
            -0.04656015195549268,
            0.2691673076247886,
            0.5759294460990402,
            0.7957073831830868,
            0.9345634482649406,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.29650053704427826,
            -0.29702139861645266,
            -0.2970738847754726,
            -0.29684227184111633,
            -0.295908233579969,
            -0.29716210543131866,
            -0.29687932234649017,
            -0.2971028442177169,
            -0.2949186735594533,
            -0.2979404887735794,
            -0.28819212808539174,
            -0.29614884627884625,
            -0.2999677106877632,
            -0.2824929178470267,
            -0.29098185491704526,
            -0.29026829952322786,
            -0.1382409968349895,
            0.16811610130521104,
            0.46322344067645704,
            0.7391787683281347,
            0.9380040446383967,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        self.0
            .iter()
            .zip(optimal)
            .map(|((v, _), v_star)| {
                let diff = v - v_star;
                diff * diff
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuboid_features() {
        assert_eq!(
            Vector::cuboid_features(
                &State {
                    player: 4,
                    dealer: 4
                },
                &Action::Stick
            )
            .w,
            vec![
                0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ]
        )
    }
}
