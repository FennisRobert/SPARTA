//! Tunes SPARTA to this computer on the standard problems and saves the result.
//!
//!     cargo run --release --example optimize [NAME] [SECONDS]
//!
//! NAME saves the result as a profile (see `sparta::profiles`); SECONDS caps
//! the search. Progress prints as it goes, then a summary.
use sparta::optimize::{optimize, Options, Problem};
use std::time::Duration;

fn main() -> sparta::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next();
    let options = Options {
        time_limit: args.next().and_then(|s| s.parse().ok()).map(Duration::from_secs_f64),
        ..Options::default()
    };
    let outcome = optimize(&Problem::standard(), &options, &mut |p| {
        eprintln!("{p}");
        true
    })?;
    println!("{outcome}");
    if let Some(name) = name {
        sparta::profiles::save(&name, &outcome.profile())?;
        println!("  Saved as '{name}' in {}", sparta::profiles::path().display());
    }
    Ok(())
}
