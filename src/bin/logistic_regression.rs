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

use humantime::format_duration;
use rayon::prelude::*;
use regex::Regex;
use rusty_machine::analysis::score::*;
use rusty_machine::learning;
use rusty_machine::linalg;
use rusty_machine::prelude::SupModel;
use std::default;
use std::io;
use std::time::Instant;

fn main() -> io::Result<()> {
    simple_logger::init().unwrap();

    info!("reading training data");
    let training_data_raw = std::include_str!("../data/train.csv");
    let mut training_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(training_data_raw.as_bytes());
    let mut tmp_training_data: Vec<rusty_titanic::TitanicTrainingData> = Vec::new();

    for record in training_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTrainingData = record?;
        tmp_training_data.push(record);
    }

    let training_data: Vec<f64> = parse_training_data(&tmp_training_data)?;

    info!("reading test data");
    let test_data_raw = std::include_str!("../data/test.csv");
    let mut test_data_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(test_data_raw.as_bytes());
    let mut tmp_test_data: Vec<rusty_titanic::TitanicTestData> = Vec::new();
    for record in test_data_reader.deserialize() {
        let record: rusty_titanic::TitanicTestData = record?;
        tmp_test_data.push(record);
    }
    let test_data: Vec<f64> = parse_test_data(&tmp_test_data)?;

    // info!(
    //     "training_data.len(): {}, training_data.len() / 8: {}",
    //     training_data.len(),
    //     training_data.len() / 8
    // );
    let inputs = linalg::Matrix::new(training_data.len() / 8, 8, training_data);
    let target_data: Vec<_> = tmp_training_data
        .iter()
        .filter(|x| !x.ticket.contains(&"LINE"))
        .map(|x| x.survived as f64)
        .collect();
    // info!("target_data.len(): {}", target_data.len());
    let targets = linalg::Vector::new(target_data);
    let mut log_mod = learning::logistic_reg::LogisticRegressor::default();
    log_mod.train(&inputs, &targets).unwrap();

    let new_point = linalg::Matrix::new(test_data.len() / 8, 8, test_data);
    let output = log_mod.predict(&new_point).unwrap();

    info!("output: {:?}", output);

    let output_matrix = linalg::Matrix::new(1, 1, output);
    let targets_matrix = linalg::Matrix::new(1, 1, targets);
    info!("row_accuracy: {}", row_accuracy(&output_matrix, &targets_matrix));
    // info!("accuracy: {}", accuracy(output.iter(), targets.iter()));

    let start = Instant::now();
    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}

fn parse_training_data(raw_training_data: &Vec<rusty_titanic::TitanicTrainingData>) -> io::Result<Vec<f64>> {
    let missing_age_count = raw_training_data.iter().filter(|x| x.age.is_none()).count();
    let age_sum: f64 = raw_training_data.iter().filter(|x| x.age.is_some()).map(|x| x.age.unwrap() as f64).sum();
    let age_mean = age_sum / ((raw_training_data.len() - missing_age_count) as f64);
    info!(
        "missing_age_count: {} out of {}, age_sum: {}, age_mean: {}",
        missing_age_count,
        raw_training_data.len(),
        age_sum,
        age_mean
    );

    let mut training_data: Vec<f64> = Vec::new();
    for record in raw_training_data.iter() {
        let mut tmp_training_data: Vec<f64> = Vec::new();
        //tmp_training_data.push(record.passenger_id.clone() as f64);
        //tmp_training_data.push(record.survived.clone() as f64);
        tmp_training_data.push(record.passenger_class.clone() as f64);

        let mut name = record.name.clone();
        // name = name.replace(&honorific, "");
        // let name_split: Vec<_> = name.split(",").collect();
        // let last_name = name_split.get(0).unwrap();
        // let first_name = name_split.get(1).unwrap();
        //info!("{} {} {}", honorific, first_name.trim(), last_name);
        // leave name out????

        match rusty_titanic::COMMON_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        }

        match rusty_titanic::MILITARY_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        }

        match rusty_titanic::FORMAL_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        }

        match rusty_titanic::ACADEMIC_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        }

        match rusty_titanic::RELIGIOUS_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        }

        match record.sex.clone().as_str() {
            "male" => tmp_training_data.push(1f64),
            _ => tmp_training_data.push(0f64),
        };

        match record.age {
            None => tmp_training_data.push(age_mean.clone()),
            _ => tmp_training_data.push(record.age.unwrap().clone() as f64),
        }

        // tmp_training_data.push(record.siblings_spouses_aboard.clone() as f64);
        // tmp_training_data.push(record.parents_children_aboard.clone() as f64);

        let ticket = record.ticket.clone();
        let ticket_regex = Regex::new(r"(SC/AH Balse |LINE|.* )?([0-9]+)?").unwrap();
        let ticket_capture_groups = ticket_regex.captures(ticket.as_str()).unwrap();
        let ticket_prefix = ticket_capture_groups.get(1).map_or("", |m| m.as_str());
        let ticket_number = ticket_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("ticket: {}: ticket_prefix: {}, ticket_number: {}", ticket, ticket_prefix, ticket_number);
        if ticket_prefix.eq("LINE") {
            continue;
        }
        // tmp_training_data.push(ticket_number.parse::<f64>().unwrap());
        // tmp_training_data.push(record.fare.clone() as f64);

        if record.cabin_number.is_some() {
            let cabin = record.cabin_number.clone().unwrap();
            let cabin_regex = Regex::new(r"([A-Z]?)([0-9]*)").unwrap();
            let cabin_capture_groups = cabin_regex.captures(cabin.as_str()).unwrap();
            let deck = cabin_capture_groups.get(1).map_or("", |m| m.as_str());
            let room = cabin_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("cabin: {}: deck: {}, room: {}", cabin, deck, room);

        // match deck {
        //     "A" => tmp_training_data.push(7f64),
        //     "B" => tmp_training_data.push(6f64),
        //     "C" => tmp_training_data.push(5f64),
        //     "D" => tmp_training_data.push(4f64),
        //     "E" => tmp_training_data.push(3f64),
        //     "F" => tmp_training_data.push(2f64),
        //     "G" => tmp_training_data.push(1f64),
        //     _ => tmp_training_data.push(0f64),
        // }

        //has multiple cabins
        // match cabin.contains(" ") {
        //     true => tmp_training_data.push(1f64),
        //     _ => tmp_training_data.push(0f64),
        // }
        } else {
            // don't know deck
            // tmp_training_data.push(0f64);
            // does not have multiple cabins
            // tmp_training_data.push(0f64);
        }

        // match record.embarked.clone().as_str() {
        //     "S" => tmp_training_data.push(1f64),
        //     "C" => tmp_training_data.push(2f64),
        //     "Q" => tmp_training_data.push(3f64),
        //     _ => tmp_training_data.push(0f64),
        // };
        // should be 17
        //info!("tmp_training_data.len(): {}", tmp_training_data.len());
        training_data.append(&mut tmp_training_data);
    }
    Ok(training_data)
}

fn parse_test_data(raw_test_data: &Vec<rusty_titanic::TitanicTestData>) -> io::Result<Vec<f64>> {
    let missing_age_count = raw_test_data.iter().filter(|x| x.age.is_none()).count();
    let age_sum: f64 = raw_test_data.iter().filter(|x| x.age.is_some()).map(|x| x.age.unwrap() as f64).sum();
    let age_mean = age_sum / ((raw_test_data.len() - missing_age_count) as f64);
    info!(
        "missing_age_count: {} out of {}, age_sum: {}, age_mean: {}",
        missing_age_count,
        raw_test_data.len(),
        age_sum,
        age_mean
    );

    let missing_fare_count = raw_test_data.iter().filter(|x| x.fare.is_none()).count();
    let fare_sum: f64 = raw_test_data.iter().filter(|x| x.fare.is_some()).map(|x| x.fare.unwrap() as f64).sum();
    let fare_mean = fare_sum / ((raw_test_data.len() - missing_fare_count) as f64);
    info!(
        "missing_fare_count: {} out of {}, fare_sum: {}, fare_mean: {}",
        missing_fare_count,
        raw_test_data.len(),
        fare_sum,
        fare_mean
    );

    let mut test_data: Vec<_> = Vec::new();
    for record in raw_test_data.iter() {
        let mut tmp_test_data: Vec<f64> = Vec::new();
        //info!("record: {}", record);
        //tmp_test_data.push(record.passenger_id.clone() as f64);
        tmp_test_data.push(record.passenger_class.clone() as f64);

        let mut name = record.name.clone();
        // name = name.replace(&honorific, "");
        // let name_split: Vec<_> = name.split(",").collect();
        // let last_name = name_split.get(0).unwrap();
        // let first_name = name_split.get(1).unwrap();
        //info!("{} {} {}", honorific, first_name.trim(), last_name);
        // leave name out????

        match rusty_titanic::COMMON_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        }

        match rusty_titanic::MILITARY_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        }

        match rusty_titanic::FORMAL_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        }

        match rusty_titanic::ACADEMIC_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        }

        match rusty_titanic::RELIGIOUS_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        }

        match record.sex.clone().as_str() {
            "male" => tmp_test_data.push(1f64),
            _ => tmp_test_data.push(0f64),
        };

        match record.age {
            None => tmp_test_data.push(age_mean.clone()),
            _ => tmp_test_data.push(record.age.unwrap().clone() as f64),
        }

        // tmp_test_data.push(record.siblings_spouses_aboard.clone() as f64);
        // tmp_test_data.push(record.parents_children_aboard.clone() as f64);

        //push ticket here
        let ticket = record.ticket.clone();
        let ticket_regex = Regex::new(r"(SC/AH Balse |LINE|.* )?([0-9]+)?").unwrap();
        let ticket_capture_groups = ticket_regex.captures(ticket.as_str()).unwrap();
        let ticket_prefix = ticket_capture_groups.get(1).map_or("", |m| m.as_str());
        let ticket_number = ticket_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("ticket: {}: ticket_prefix: {}, ticket_number: {}", ticket, ticket_prefix, ticket_number);
        if ticket_prefix.eq("LINE") {
            continue;
        }

        // tmp_test_data.push(ticket_number.parse::<f64>().unwrap());
        //
        // match record.fare {
        //     None => tmp_test_data.push(fare_mean.clone()),
        //     _ => tmp_test_data.push(record.fare.unwrap().clone() as f64),
        // }

        if record.cabin_number.is_some() {
            let cabin = record.cabin_number.clone().unwrap();
            let cabin_regex = Regex::new(r"([A-Z]?)([0-9]*)").unwrap();
            let cabin_capture_groups = cabin_regex.captures(cabin.as_str()).unwrap();
            let deck = cabin_capture_groups.get(1).map_or("", |m| m.as_str());
            let room = cabin_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("cabin: {}: deck: {}, room: {}", cabin, deck, room);

        // match deck {
        //     "A" => tmp_test_data.push(7f64),
        //     "B" => tmp_test_data.push(6f64),
        //     "C" => tmp_test_data.push(5f64),
        //     "D" => tmp_test_data.push(4f64),
        //     "E" => tmp_test_data.push(3f64),
        //     "F" => tmp_test_data.push(2f64),
        //     "G" => tmp_test_data.push(1f64),
        //     _ => tmp_test_data.push(0f64),
        // }

        //has multiple cabins
        // match cabin.contains(" ") {
        //     true => tmp_test_data.push(1f64),
        //     _ => tmp_test_data.push(0f64),
        // }
        } else {
            // don't know deck
            // tmp_test_data.push(0f64);
            // does not have multiple cabins
            // tmp_test_data.push(0f64);
        }

        // match record.embarked.clone().as_str() {
        //     "S" => tmp_test_data.push(1f64),
        //     "C" => tmp_test_data.push(2f64),
        //     "Q" => tmp_test_data.push(3f64),
        //     _ => tmp_test_data.push(0f64),
        // };

        // should be 16
        //info!("tmp_test_data.len(): {}", tmp_test_data.len());
        test_data.append(&mut tmp_test_data);
    }
    Ok(test_data)
}
