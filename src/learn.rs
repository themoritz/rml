use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use rand::{prelude::ThreadRng, thread_rng};
use std::thread;

pub enum Req<S> {
    GetState,
    SetState { state: S },
}

pub fn queryable_state<S, F>(mut state: S, update: F) -> (SyncSender<Req<S>>, Receiver<S>)
where
    S: Clone + Send + 'static,
    F: Fn(&mut ThreadRng, &mut S) + Send + 'static,
{
    let (req_s, req_r) = sync_channel(2);
    let (resp_s, resp_r) = sync_channel(1);

    thread::spawn(move || {
        let mut rng = thread_rng();
        loop {
            update(&mut rng, &mut state);

            // Send state to response channel if requested.
            if let Ok(req) = req_r.try_recv() {
                match req {
                    Req::GetState => {
                        if let Err(TrySendError::Disconnected(_)) = resp_s.try_send(state.clone()) {
                            panic!("Could not respond state: Disconnected")
                        }
                    }
                    Req::SetState { state: new_state } => state = new_state,
                }
            }
        }
    });

    (req_s, resp_r)
}
