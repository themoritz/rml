use crossbeam::{channel, Receiver, Sender, TrySendError};
use std::thread;

pub fn queryable_state<S, F>(mut state: S, update: F) -> (Sender<()>, Receiver<S>)
where
    S: Clone + Send + 'static,
    F: Fn(&mut S) + Send + 'static,
{
    let (req_s, req_r) = channel::bounded(1);
    let (resp_s, resp_r) = channel::bounded(1);

    thread::spawn(move || {
        loop {
            update(&mut state);

            // Send state to response channel if requested.
            if let Ok(_) = req_r.try_recv() {
                if let Err(TrySendError::Disconnected(_)) = resp_s.try_send(state.clone()) {
                    panic!("Could not respond state: Disconnected")
                }
                // We don't want requests to pile up so we empty the queue after responding.
                for _ in req_r.try_iter() {}
            }
        }
    });

    (req_s, resp_r)
}
