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
use rusty_machine::learning::nnet::{BCECriterion, NeuralNet};
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "titanic_nnet", about = "")]
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

    let raw_data = std::include_str!("../data/train.csv");
    let mut data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_data.as_bytes());
    let mut training_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();
    for record in data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        training_data.push(record);
    }
    debug!("training_data.len(): {}", training_data.len());

    let (training_data_matrix, training_targets, training_feature_size) = rusty_titanic::parse_training_data(&training_data)?;
    let training_targets_matrix = linalg::Matrix::new(training_targets.data().len(), 1, training_targets);

    let test_data_raw = std::include_str!("../data/test.csv");
    let mut test_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(test_data_raw.as_bytes());
    let mut tmp_test_data: Vec<rusty_titanic::TitanicTestData> = Vec::new();
    for record in test_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTestData = record?;
        tmp_test_data.push(record);
    }

    let criterion = BCECriterion::new(Regularization::L2(0.1f64));
    //let mut model = NeuralNet::new(&[13, 121, 1], criterion, StochasticGD::default());
    let experimental_layers = vec![vec![training_feature_size, training_feature_size , 1]];
    let layers = vec![training_feature_size, 121, 1];
    let mut model = NeuralNet::new(layers.as_ref(), criterion, StochasticGD::default());
    let accuracy_per_fold: Vec<f64> = k_fold_validate(
        &mut model,
        &training_data_matrix,
        &training_targets_matrix,
        10,
        rusty_titanic::rounded_row_accuracy,
    )
    .unwrap();
    info!("accuracy_per_fold: {:?}", accuracy_per_fold);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
