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
use rusty_machine::learning::svm::SVM;
use rusty_machine::learning::SupModel;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "svm_accuracy", about = "")]
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
    let (training_targets, training_data_matrix) = rusty_titanic::parse_training_data(&training_data)?;
    let fixed_training_targets = training_targets.apply(&fix);

    let test_data = training_data_split.1.to_vec();
    debug!("test_data.len(): {}", test_data.len());
    let (test_targets, test_data_matrix) = rusty_titanic::parse_training_data(&test_data)?;
    let fixed_test_targets = test_targets.apply(&fix);
    debug!("fixed_test_targets: {:?}", fixed_test_targets);

    let mut svm_mod = SVM::default();
    svm_mod.optim_iters = 10000;
    svm_mod.train(&training_data_matrix, &fixed_training_targets).unwrap();
    let outputs = svm_mod.predict(&test_data_matrix).unwrap();
    debug!("outputs: {:?}", outputs);

    let acc = accuracy(outputs.iter(), fixed_test_targets.iter());
    info!("accuracy: {}", acc);

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
