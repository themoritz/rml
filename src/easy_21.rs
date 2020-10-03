use rand::distributions::{uniform::Uniform, Distribution};
use rand::Rng;
use std::collections::HashMap;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct State {
    dealer: i32,
    player: i32,
}

impl State {
    pub fn init<R: Rng>(rng: &mut R) -> Self {
        Self {
            dealer: Card::draw(rng).value,
            player: Card::draw(rng).value,
        }
    }
}

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
                Color::Black
            } else {
                Color::Red
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
pub struct V(HashMap<State, f32>);

impl V {
    fn get(&self, state: &State) -> f32 {
        self.0.get(state).map_or(0.0, |v| *v)
    }

    fn set(&mut self, state: &State, v: f32) -> &mut Self {
        self.0.insert(*state, v);
        self
    }
}
