// config/hot_reload.rs
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc::channel;
use std::time::Duration;
use std::path::PathBuf;

pub fn watch_config(path: PathBuf, on_reload: impl Fn() + Send + 'static) {
    let (tx, rx) = channel();

    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |res| {
        if let Ok(event) = res {
            if event.kind.is_modify() {
                on_reload();
            }
        }
    }).unwrap();

    watcher.watch(&path, RecursiveMode::NonRecursive).unwrap();

    std::thread::spawn(move || {
        loop {
            if let Ok(_) = rx.recv_timeout(Duration::from_secs(1)) {
                // Already triggered in callback
            }
        }
    });
}
