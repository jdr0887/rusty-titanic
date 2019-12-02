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
    let (training_data_matrix, training_targets, _) = rusty_titanic::parse_training_data(&training_data)?;
    let training_targets_matrix = linalg::Matrix::new(training_targets.data().len(), 1, training_targets);
    // debug!("training_targets_matrix: {:?}", training_targets_matrix);

    let mut model = NaiveBayes::<Bernoulli>::new();
    let accuracy_per_fold: Vec<f64> = k_fold_validate(&mut model, &training_data_matrix, &training_targets_matrix, 10, row_accuracy).unwrap();
    info!("accuracy_per_fold: {:?}", accuracy_per_fold);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
