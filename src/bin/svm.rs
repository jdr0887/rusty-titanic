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

use humantime::format_duration;
use log::Level;
use rusty_machine::learning::svm::SVM;
use rusty_machine::learning::SupModel;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "svm", about = "")]
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

    let raw_training_data = std::include_str!("../data/train.csv");
    let mut training_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_training_data.as_bytes());
    let mut training_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();
    for record in training_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        training_data.push(record);
    }
    let (training_data_matrix, training_targets) = rusty_titanic::parse_training_data(&training_data)?;
    debug!("training_data_matrix: {:?}", training_data_matrix);
    let fixed_training_targets = training_targets.apply(&fix);
    debug!("fixed_training_targets: {:?}", fixed_training_targets);

    let raw_test_data = std::include_str!("../data/test.csv");
    let mut test_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_test_data.as_bytes());
    let mut test_data: Vec<rusty_titanic::TitanicTestData> = Vec::new();
    for record in test_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTestData = record?;
        test_data.push(record);
    }
    let test_data_matrix = rusty_titanic::parse_test_data(&test_data)?;
    debug!("test_data_matrix: {:?}", test_data_matrix);

    let mut svm_mod = SVM::default();
    svm_mod.optim_iters = 10000;
    svm_mod.train(&training_data_matrix, &fixed_training_targets).unwrap();
    let outputs = svm_mod.predict(&test_data_matrix).unwrap();
    info!("outputs: {:?}", outputs);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}

fn fix(a: f64) -> f64 {
    if a == 0f64 {
        -1f64
    } else {
        a
    }
}
