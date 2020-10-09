use rand::distributions::{uniform::Uniform, Distribution};
use rand::Rng;

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

enum Color {
    Black,
    Red,
}

struct Card {
    value: i32,
    color: Color,
}

impl Card {
    fn add_to(&self, x: i32) -> i32 {
        x + self.value
            * match self.color {
                Color::Black => 1,
                Color::Red => -1,
            }
    }

    fn draw<R: Rng>(rng: &mut R) -> Self {
        let v = Uniform::new_inclusive(1, 10);
        Self {
            value: v.sample(rng),
            color: if rng.gen::<f64>() < (1.0 / 3.0) {
                Color::Red
            } else {
                Color::Black
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
    x < 1 || x > 21
}

fn signum(x: i32) -> i32 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
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

pub fn example_policy<R: Rng>(rng: &mut R, state: &State) -> Action {
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
}

impl MCState {
    pub fn init() -> Self {
        Self {
            v: V::init((0.0, 0)),
            episodes: 0,
        }
    }
}

impl HasV for MCState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
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

fn epsilon_greedy<R: Rng>(rng: &mut R, eps: f64, q: &Q<(f64, i32)>, state: &State) -> Action {
    let q_hit = q.get(state, &Action::Hit).0;
    let q_stick = q.get(state, &Action::Stick).0;
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
        mc_state.v.update(&state, |(_, n)| (q_hit.max(q_stick), n + 1));
    }
    // Inc episodes
    mc_state.episodes += 1;
}

#[derive(Clone)]
pub struct TDState {
    pub v: V<(f64, i32)>,
    pub eligibility_traces: V<f64>,
    pub episodes: i32,
}

impl TDState {
    pub fn init() -> Self {
        Self {
            v: V::init((0.0, 0)),
            eligibility_traces: V::init(0.0),
            episodes: 0,
        }
    }
}

impl HasV for TDState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
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
}

impl TDControlState {
    pub fn init() -> Self {
        Self {
            v: V::init((0.0, 0)),
            q: Q::init((0.0, 0)),
            eligibility_traces: Q::init(0.0),
            episodes: 0,
        }
    }
}

impl HasV for TDControlState {
    fn get_v(&self, state: &State) -> f64 {
        self.v.get(state).0
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
        td_state.eligibility_traces.update(&state, &action, |v| v + 1.0);

        let td_error =
            (sample.reward as f64) + td_state.q.get(&next_state, &next_action).0 - td_state.q.get(&state, &action).0;
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
        td_state.v.update(&state, |(_, n)| (q_hit.max(q_stick), n + 1));

        if sample.terminal {
            break;
        } else {
            state = next_state;
            action = next_action;
        }
    }
    td_state.episodes += 1;
}
