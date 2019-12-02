#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate itertools;
extern crate rayon;
extern crate regex;
extern crate rustlearn;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;
extern crate libm;

use humantime::format_duration;
use log::Level;
use rusty_machine::analysis::cross_validation::k_fold_validate;
use rusty_machine::analysis::score::row_accuracy;
use rusty_machine::learning::naive_bayes::{Bernoulli, NaiveBayes};
use rusty_machine::learning::SupModel;
use rusty_machine::linalg;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "scratch", about = "")]
struct Options {
    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "info")]
    log_level: String,
}
fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    info!("a: {}", libm::round(1.325f64));
    info!("b: {}", libm::round(1.365f64));
    info!("c: {}", libm::roundf(1.25f32));
    info!("d: {}", libm::roundf(1.65f32));

    

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
