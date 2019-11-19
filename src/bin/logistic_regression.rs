#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use rusty_machine::learning;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use rusty_machine::prelude::SupModel;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "logistic_regression", about = "")]
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
    let raw_training_data = std::include_str!("../data/train.csv");
    let mut training_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_training_data.as_bytes());
    let mut training_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();
    for record in training_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        training_data.push(record);
    }
    debug!("training_data.len(): {}", training_data.len());

    let (training_targets, training_data_matrix) = rusty_titanic::parse_training_data(&training_data)?;

    debug!("reading test data");
    let test_data_raw = std::include_str!("../data/test.csv");
    let mut test_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(test_data_raw.as_bytes());
    let mut test_data: Vec<rusty_titanic::TitanicTestData> = Vec::new();
    for record in test_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTestData = record?;
        test_data.push(record);
    }
    debug!("test_data.len(): {}", test_data.len());

    let test_data_matrix = rusty_titanic::parse_test_data(&test_data)?;

    let gradient_desc = GradientDesc::new(0.01, 4000);
    let mut model = learning::logistic_reg::LogisticRegressor::new(gradient_desc);
    model.train(&training_data_matrix, &training_targets).unwrap();
    let outputs = model.predict(&test_data_matrix).unwrap();
    debug!("outputs: {:?}", outputs);
    let rounded_outputs = outputs.apply(&round);
    info!("rounded_outputs: {:?}", rounded_outputs);

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