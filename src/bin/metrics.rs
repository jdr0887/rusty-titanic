#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate itertools;
extern crate libm;
extern crate rayon;
extern crate regex;
extern crate rustlearn;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use itertools::Itertools;
use log::Level;
use rusty_machine::analysis::cross_validation::k_fold_validate;
use rusty_machine::analysis::score::row_accuracy;
use rusty_machine::learning::naive_bayes::{Bernoulli, NaiveBayes};
use rusty_machine::learning::SupModel;
use rusty_machine::linalg;
use std::collections;
use std::io;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "split_name", about = "")]
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

    let raw_data = std::include_str!("../data/train.csv");
    let mut data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(raw_data.as_bytes());
    let mut training_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();
    for record in data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        training_data.push(record);
    }
    debug!("training_data.len(): {}", training_data.len());

    let missing_age_count = training_data.iter().filter(|x| x.age.is_none()).count();
    let age_sum: f64 = training_data.iter().filter(|x| x.age.is_some()).map(|x| x.age.unwrap() as f64).sum();
    let age_mean = age_sum / ((training_data.len() - missing_age_count) as f64);
    info!(
        "missing_age_count: {} out of {}, age_mean: {}",
        missing_age_count,
        training_data.len(),
        age_mean
    );

    let missing_fare_count = training_data.iter().filter(|x| x.fare.is_none()).count();
    let fare_sum: f64 = training_data.iter().filter(|x| x.fare.is_some()).map(|x| x.fare.unwrap() as f64).sum();
    let fare_mean = fare_sum / ((training_data.len() - missing_fare_count) as f64);
    info!(
        "missing_fare_count: {} out of {}, fare_mean: {}",
        missing_fare_count,
        training_data.len(),
        fare_mean
    );

    let rounded_fare_groups: collections::HashMap<i32, Vec<f32>> = training_data
        .iter()
        .map(|x| (libm::roundf(x.fare.unwrap()) as i32, x.fare.unwrap()))
        .into_group_map();

    for (k, v) in rounded_fare_groups.iter() {
        info!("rounded_fare_groups...key: {}, count: {}", k, v.len());
    }

    let embarked_groups: collections::HashMap<String, Vec<String>> = training_data
        .iter()
        .map(|x| (x.embarked.to_string(), x.embarked.to_string()))
        .into_group_map();

    for (k, v) in embarked_groups.iter() {
        info!("embarked from: {}, count: {}", k, v.len());
    }

    let age_ranges = vec![
        (0f32..=3f32),
        (4f32..=7f32),
        (8f32..=11f32),
        (12f32..=15f32),
        (16f32..=19f32),
        (20f32..=30f32),
        (31f32..=40f32),
        (41f32..=50f32),
        (51f32..=60f32),
        (61f32..=70f32),
        (71f32..=80f32),
    ];

    for age_range in age_ranges {
        let age_filtered: Vec<_> = training_data.iter().filter(|a| a.age.is_some()).collect();

        info!(
            "age range: {:?} & Master: {}, Miss: {}, Ms: {}, Mrs: {}, Mr: {}",
            age_range,
            age_filtered
                .iter()
                .filter(|a| a.name.contains(&"Master.") && age_range.contains(&a.age.unwrap()))
                .count(),
            age_filtered
                .iter()
                .filter(|a| a.name.contains(&"Miss.") && age_range.contains(&a.age.unwrap()))
                .count(),
            age_filtered
                .iter()
                .filter(|a| a.name.contains(&"Ms.") && age_range.contains(&a.age.unwrap()))
                .count(),
            age_filtered
                .iter()
                .filter(|a| a.name.contains(&"Mrs.") && age_range.contains(&a.age.unwrap()))
                .count(),
            age_filtered
                .iter()
                .filter(|a| a.name.contains(&"Mr.") && age_range.contains(&a.age.unwrap()))
                .count()
        );
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
