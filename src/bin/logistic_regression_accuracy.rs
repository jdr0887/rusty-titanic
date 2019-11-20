#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use rusty_machine::analysis::score::*;
use rusty_machine::learning;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use rusty_machine::learning::SupModel;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "logistic_regression_accuracy", about = "")]
struct Options {
    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "Info")]
    log_level: String,
}
fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    debug!("reading training data");
    let raw_data = std::include_str!("../data/train.csv");
    let mut data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_data.as_bytes());
    let mut all_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();
    for record in data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        all_data.push(record);
    }

    let split_idx = (all_data.len() as f32 * 0.7) as usize;

    let training_data_split = all_data.split_at_mut(split_idx);

    let training_data = training_data_split.0.to_vec();
    debug!("training_data.len(): {}", training_data.len());
    let (training_data_matrix, training_targets) = rusty_titanic::parse_training_data(&training_data)?;

    let test_data = training_data_split.1.to_vec();
    debug!("test_data.len(): {}", test_data.len());
    let (test_data_matrix, test_targets) = rusty_titanic::parse_training_data(&test_data)?;
    debug!("test_targets: {:?}", test_targets);

    let gradient_desc = GradientDesc::new(0.01, 4000);
    let mut model = learning::logistic_reg::LogisticRegressor::new(gradient_desc);
    model.train(&training_data_matrix, &training_targets).unwrap();
    let outputs = model.predict(&test_data_matrix).unwrap();
    debug!("outputs: {:?}", outputs);

    let rounded_outputs = outputs.apply(&round);
    debug!("rounded_outputs: {:?}", rounded_outputs);
    let acc = accuracy(rounded_outputs.iter(), test_targets.iter());
    info!("accuracy: {}", acc);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}

fn round(a: f64) -> f64 {
    if a > 0.5 {
        1.0f64
    } else {
        0f64
    }
}
