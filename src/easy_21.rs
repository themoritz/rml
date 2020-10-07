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
            color: if rng.gen::<f32>() < (1.0 / 3.0) {
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

#[derive(Clone)]
pub struct V<T>(Vec<T>);

impl<T> V<T> {
    #[inline]
    fn index(state: &State) -> usize {
        (21 * state.player + state.dealer) as usize
    }

    pub fn init(v: T) -> Self
    where
        T: Clone,
    {
        V(vec![v; 21 * 21 + 21])
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

    fn transform<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        for v in self.0.iter_mut() {
            *v = f(v);
        }
        self
    }
}

#[derive(Clone)]
pub struct Q<T>(Vec<T>);

impl<T> Q<T> {
    #[inline]
    fn index(state: &State, action: &Action) -> usize {
        (21 * state.player + state.dealer) as usize
            + match action {
                Action::Stick => 21 * 21 + 21,
                Action::Hit => 0,
            }
    }

    pub fn init(v: T) -> Self
    where
        T: Clone,
    {
        Q(vec![v; 2 * (21 * 21 + 21)])
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

pub fn monte_carlo_prediction<R: Rng, P: Fn(&mut R, &State) -> Action>(
    rng: &mut R,
    policy: P,
    v: &mut V<(f32, i32)>,
) {
    let (state_actions, reward) = episode(rng, policy);
    for (state, _) in state_actions {
        let (value, n) = v.get(&state);
        let new_n = n + 1;
        let new_value = value + 1.0 / (new_n as f32) * (reward as f32 - value);
        v.set(&state, (new_value, new_n));
    }
}

fn epsilon_greedy<R: Rng>(rng: &mut R, eps: f32, q: &Q<(f32, i32)>, state: &State) -> Action {
    let q_hit = q.get(state, &Action::Hit).0;
    let q_stick = q.get(state, &Action::Stick).0;
    let threshold = if q_hit > q_stick {
        eps / 2.0
    } else {
        1.0 - eps / 2.0
    };
    if rng.gen::<f32>() < threshold {
        Action::Stick
    } else {
        Action::Hit
    }
}

fn greedy_episode<R: Rng>(rng: &mut R, mc_state: &MCState) -> (Vec<(State, Action)>, i32) {
    episode(rng, |rng, state| {
        let n_0 = 10_000.0;
        let eps = n_0 / (n_0 + mc_state.v.get(state).1 as f32);
        epsilon_greedy(rng, eps, &mc_state.q, state)
    })
}

#[derive(Clone)]
pub struct MCState {
    pub v: V<(f32, i32)>,
    pub q: Q<(f32, i32)>,
    pub episodes: i32,
}

impl MCState {
    pub fn init() -> Self {
        MCState {
            v: V::init((0.0, 0)),
            q: Q::init((0.0, 0)),
            episodes: 0,
        }
    }
}

pub fn monte_carlo_control<R: Rng>(rng: &mut R, mc_state: &mut MCState) {
    let (state_actions, reward) = greedy_episode(rng, mc_state);
    for (state, action) in state_actions {
        // Update Q
        let (value, n) = mc_state.q.get(&state, &action);
        let new_n = n + 1;
        let new_value = value + 1.0 / (new_n as f32) * (reward as f32 - value);
        mc_state.q.set(&state, &action, (new_value, new_n));

        // Update V
        let (_, n) = mc_state.v.get(&state);
        let q_hit = mc_state.q.get(&state, &Action::Hit).0;
        let q_stick = mc_state.q.get(&state, &Action::Stick).0;
        mc_state.v.set(&state, (q_hit.max(q_stick), n + 1));
    }
    // Inc episodes
    mc_state.episodes += 1;
}

pub struct TDState {
    v: V<f32>,
    eligibility_traces: V<f32>,
}

/// One episode, will be looped over by the main loop.
pub fn td_lambda_prediction<R: Rng, P: Fn(&mut R, &State) -> Action>(
    rng: &mut R,
    lambda: f32,
    policy: P,
    td_state: &mut TDState,
) {
    let mut state = State::init(rng);
    loop {
        let action = policy(rng, &state);
        let sample = step(rng, state, action);

        let next_state = sample.state;

        // Update eligibility traces
        td_state.eligibility_traces.transform(|v| v * lambda);
        td_state.eligibility_traces.set(&state, td_state.eligibility_traces.get(&state) + 1.0);

        if sample.terminal {
            break;
        }
    }
}
