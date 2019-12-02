#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate itertools;
extern crate rayon;
extern crate regex;
extern crate rulinalg;
extern crate rustlearn;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use rusty_machine::analysis::cross_validation::k_fold_validate;
use rusty_machine::learning::nnet::{BCECriterion, NeuralNet};
use rusty_machine::learning::optim::grad_desc::*;
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
    let (training_data_matrix, training_targets, training_feature_size) = rusty_titanic::parse_training_data(&training_data)?;
    debug!("training_targets: {:?}", training_targets);
    let training_targets_matrix = linalg::Matrix::new(training_targets.data().len(), 1, training_targets);

    //let mut model = NeuralNet::new(&[13, 121, 1], criterion, StochasticGD::default());

    let mut layers: Vec<Vec<usize>> = Vec::new();

    let mut no_hidden_layers = vec![vec![training_feature_size, 1]];

    let mut one_hidden_layers = vec![
        vec![training_feature_size, training_feature_size * 4, 1],
        vec![training_feature_size, training_feature_size * 8, 1],
        vec![training_feature_size, training_feature_size * 12, 1],
    ];

    let mut two_hidden_layers = vec![
        vec![training_feature_size, training_feature_size * 6, training_feature_size * 3, 1],
        vec![training_feature_size, training_feature_size * 10, training_feature_size * 5, 1],
    ];

    let mut three_hidden_layers = vec![vec![
        training_feature_size,
        training_feature_size * 12,
        training_feature_size * 8,
        training_feature_size * 4,
        1,
    ]];

    layers.append(&mut no_hidden_layers);
    layers.append(&mut one_hidden_layers);
    layers.append(&mut two_hidden_layers);
    layers.append(&mut three_hidden_layers);

    for layer in layers.iter() {
        //let mut model = NeuralNet::new(layer.as_ref(), BCECriterion::new(Regularization::L2(0.01)), StochasticGD::default());
        //let mut model = NeuralNet::new(layer.as_ref(), BCECriterion::new(Regularization::L1(0.05)), StochasticGD::default());
        let mut model = NeuralNet::new(layer.as_ref(), BCECriterion::default(), StochasticGD::default());
        let accuracy_per_fold: Vec<f64> = k_fold_validate(
            &mut model,
            &training_data_matrix,
            &training_targets_matrix,
            10,
            rusty_titanic::rounded_row_accuracy,
        )
        .unwrap();
        let accuracy_per_fold_sum: f64 = accuracy_per_fold.iter().sum();
        let accuracy_per_fold_mean = accuracy_per_fold_sum / accuracy_per_fold.len() as f64;
        debug!("accuracy_per_fold: {:?}", accuracy_per_fold);
        info!("layer: {:?}, accuracy_per_fold_mean: {:?}", layer, accuracy_per_fold_mean);
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
