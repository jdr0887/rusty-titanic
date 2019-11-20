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
use rusty_machine::analysis::score::*;
use rusty_machine::learning::nnet::{BCECriterion, MSECriterion, NeuralNet};
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

    // let mut training_targets_inflated: Vec<f64> = Vec::new();
    //
    // for entry in training_targets.data().iter() {
    //     if entry == &1f64 {
    //         training_targets_inflated.push(1f64);
    //         training_targets_inflated.push(0f64)
    //     } else {
    //         training_targets_inflated.push(0f64);
    //         training_targets_inflated.push(1f64)
    //     }
    // }

    let training_targets_matrix = linalg::Matrix::new(training_targets.data().len(), 1, training_targets);

    let test_data = training_data_split.1.to_vec();
    debug!("test_data.len(): {}", test_data.len());
    let (test_data_matrix, test_targets) = rusty_titanic::parse_training_data(&test_data)?;
    debug!("test_targets: {:?}", test_targets);

    // let mut test_targets_inflated: Vec<f64> = Vec::new();
    //
    // for entry in test_targets.data().iter() {
    //     if entry == &1f64 {
    //         test_targets_inflated.push(1f64);
    //         test_targets_inflated.push(0f64)
    //     } else {
    //         test_targets_inflated.push(0f64);
    //         test_targets_inflated.push(1f64)
    //     }
    // }
    // let test_targets_matrix = linalg::Matrix::new(test_targets_inflated.len() / 2, 2, test_targets_inflated);

    let test_targets_matrix = linalg::Matrix::new(test_targets.data().len(), 1, test_targets);
    debug!("test_targets_matrix: {:?}", test_targets_matrix);

    let layers = &[7, 39, 1];
    let criterion = MSECriterion::new(Regularization::L2(0.1f64));
    //let criterion = BCECriterion::new(Regularization::L2(0.1));
    let mut model = NeuralNet::new(layers, criterion, StochasticGD::default());
    model.train(&training_data_matrix, &training_targets_matrix).unwrap();
    let outputs = model.predict(&test_data_matrix).unwrap();
    debug!("outputs: {:?}", outputs);

    let mut rounded_outputs: Vec<f64> = Vec::new();
    for output in outputs.data().iter() {
        rounded_outputs.push(rusty_titanic::round(output.clone()));
    }
    debug!("rounded_outputs: {:?}", rounded_outputs);

    let rounded_outputs_matrix = linalg::Matrix::new(rounded_outputs.len(), 1, rounded_outputs);
    let row_acc = row_accuracy(&rounded_outputs_matrix, &test_targets_matrix);
    info!("row_accuracy: {}", row_acc);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
