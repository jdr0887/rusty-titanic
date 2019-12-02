#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;
extern crate libm;
extern crate rayon;
extern crate regex;
extern crate rusty_machine;

use rayon::prelude::*;
use regex::Regex;
use rusty_machine::analysis::score::accuracy;
use rusty_machine::data::transforms::{Standardizer, Transformer};
use rusty_machine::linalg;
use rusty_machine::prelude::*;
use std::fmt;
use std::io;

pub fn buffer_size() -> usize {
    2_usize.pow(14)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanicTrainingData {
    #[serde(rename = "PassengerId")]
    pub passenger_id: i32,

    #[serde(rename = "Survived")]
    pub survived: i32,

    #[serde(rename = "Pclass")]
    pub passenger_class: i32,

    #[serde(rename = "Name")]
    pub name: String,

    #[serde(rename = "Sex")]
    pub sex: String,

    #[serde(rename = "Age")]
    pub age: Option<f32>,

    #[serde(rename = "SibSp")]
    pub siblings_spouses_aboard: i32,

    #[serde(rename = "Parch")]
    pub parents_children_aboard: i32,

    #[serde(rename = "Ticket")]
    pub ticket: String,

    #[serde(rename = "Fare")]
    pub fare: Option<f32>,

    #[serde(rename = "Cabin")]
    pub cabin_number: Option<String>,

    #[serde(rename = "Embarked")]
    pub embarked: String,
}

impl fmt::Display for TitanicTrainingData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TitanicTrainingData(passenger_id: {}, survived: {}, passenger_class: {}, name: {}, sex: {}, age: {:?}, siblings_spouses_aboard: {}, parents_children_aboard: {}, ticket: {}, fare: {:?}, cabin_number: {:?}, embarked: {})",
            self.passenger_id, self.survived, self.passenger_class, self.name, self.sex, self.age, self.siblings_spouses_aboard, self.parents_children_aboard, self.ticket, self.fare, self.cabin_number, self.embarked
        )
    }
}

impl TitanicTrainingData {
    pub fn new(
        passenger_id: i32,
        survived: i32,
        passenger_class: i32,
        name: String,
        sex: String,
        age: Option<f32>,
        siblings_spouses_aboard: i32,
        parents_children_aboard: i32,
        ticket: String,
        fare: Option<f32>,
        cabin_number: Option<String>,
        embarked: String,
    ) -> TitanicTrainingData {
        TitanicTrainingData {
            passenger_id: passenger_id,
            survived: survived,
            passenger_class: passenger_class,
            name: name,
            sex: sex,
            age: age,
            siblings_spouses_aboard: siblings_spouses_aboard,
            parents_children_aboard: parents_children_aboard,
            ticket: ticket,
            fare: fare,
            cabin_number: cabin_number,
            embarked: embarked,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TitanicTestData {
    #[serde(rename = "PassengerId")]
    pub passenger_id: i32,

    #[serde(rename = "Pclass")]
    pub passenger_class: i32,

    #[serde(rename = "Name")]
    pub name: String,

    #[serde(rename = "Sex")]
    pub sex: String,

    #[serde(rename = "Age")]
    pub age: Option<f32>,

    #[serde(rename = "SibSp")]
    pub siblings_spouses_aboard: i32,

    #[serde(rename = "Parch")]
    pub parents_children_aboard: i32,

    #[serde(rename = "Ticket")]
    pub ticket: String,

    #[serde(rename = "Fare")]
    pub fare: Option<f32>,

    #[serde(rename = "Cabin")]
    pub cabin_number: Option<String>,

    #[serde(rename = "Embarked")]
    pub embarked: String,
}

impl fmt::Display for TitanicTestData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TitanicTestData(passenger_id: {}, passenger_class: {}, name: {}, sex: {}, age: {:?}, siblings_spouses_aboard: {}, parents_children_aboard: {}, ticket: {}, fare: {:?}, cabin_number: {:?}, embarked: {})",
            self.passenger_id, self.passenger_class, self.name, self.sex, self.age, self.siblings_spouses_aboard, self.parents_children_aboard, self.ticket, self.fare, self.cabin_number, self.embarked
        )
    }
}

impl TitanicTestData {
    pub fn new(
        passenger_id: i32,
        passenger_class: i32,
        name: String,
        sex: String,
        age: Option<f32>,
        siblings_spouses_aboard: i32,
        parents_children_aboard: i32,
        ticket: String,
        fare: Option<f32>,
        cabin_number: Option<String>,
        embarked: String,
    ) -> TitanicTestData {
        TitanicTestData {
            passenger_id: passenger_id,
            passenger_class: passenger_class,
            name: name,
            sex: sex,
            age: age,
            siblings_spouses_aboard: siblings_spouses_aboard,
            parents_children_aboard: parents_children_aboard,
            ticket: ticket,
            fare: fare,
            cabin_number: cabin_number,
            embarked: embarked,
        }
    }
}

lazy_static! {
    pub static ref CHILDREN_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Master.");
        ret.push("Miss.");
        ret.push("Jonkheer.");
        ret
    };
    pub static ref MALE_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Mr.");
        ret.push("Major.");
        ret.push("Capt.");
        ret.push("Col.");
        ret.push("Don.");
        ret.push("Sir.");
        ret
    };
    pub static ref FEMALE_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Mrs.");
        ret.push("Miss.");
        ret.push("Ms.");
        ret.push("Mme.");
        ret.push("Lady.");
        ret.push("Mlle.");
        ret.push("the Countess. of");
        ret
    };
    pub static ref COMMON_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Mr.");
        ret.push("Mrs.");
        ret.push("Miss.");
        ret.push("Ms.");
        ret
    };
    pub static ref MILITARY_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Major.");
        ret.push("Capt.");
        ret.push("Col.");
        ret
    };
    pub static ref FORMAL_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Don.");
        ret.push("Mme.");
        ret.push("Lady.");
        ret.push("Sir.");
        ret.push("Mlle.");
        ret.push("the Countess. of");
        ret
    };
    pub static ref ACADEMIC_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Dr.");
        ret
    };
    pub static ref RELIGIOUS_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Rev.");
        ret
    };
}

pub fn parse_training_data(
    raw_training_data: &Vec<TitanicTrainingData>,
) -> io::Result<(rusty_machine::prelude::Matrix<f64>, linalg::Vector<f64>, usize)> {
    let missing_age_count = raw_training_data.iter().filter(|x| x.age.is_none()).count();
    let age_sum: f64 = raw_training_data.iter().filter(|x| x.age.is_some()).map(|x| x.age.unwrap() as f64).sum();
    let age_mean = age_sum / ((raw_training_data.len() - missing_age_count) as f64);
    debug!(
        "missing_age_count: {} out of {}, age_sum: {}, age_mean: {}",
        missing_age_count,
        raw_training_data.len(),
        age_sum,
        age_mean
    );

    let mut feature_size = 0_usize;
    let mut targets: Vec<f64> = Vec::new();
    let mut training_data: Vec<f64> = Vec::new();
    for record in raw_training_data.iter() {
        let mut features: Vec<f64> = Vec::new();

        let ticket = record.ticket.clone();
        let ticket_regex = Regex::new(r"(SC/AH Balse |LINE|.* )?([0-9]+)?").unwrap();
        let ticket_capture_groups = ticket_regex.captures(ticket.as_str()).unwrap();
        let ticket_prefix = ticket_capture_groups.get(1).map_or("", |m| m.as_str());
        let ticket_number = ticket_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("ticket: {}: ticket_prefix: {}, ticket_number: {}", ticket, ticket_prefix, ticket_number);
        if ticket_prefix.eq("LINE") {
            continue;
        }

        //tmp_training_data.push(record.passenger_id.clone() as f64);
        targets.push(record.survived.clone() as f64);
        features.push(record.passenger_class.clone() as f64);

        let name = record.name.clone();
        // name = name.replace(&honorific, "");
        // let name_split: Vec<_> = name.split(",").collect();
        // let last_name = name_split.get(0).unwrap();
        // let first_name = name_split.get(1).unwrap();
        //info!("{} {} {}", honorific, first_name.trim(), last_name);
        // leave name out????

        let age = match record.age {
            None => age_mean.clone(),
            _ => record.age.unwrap().clone() as f64,
        };

        features.push(age);
        features.push(record.passenger_class.clone() as f64 * age);

        match (0f64..=16f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (17f64..=32f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (33f64..=48f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (49f64..=99f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (0f64..=16f64).contains(&age) && CHILDREN_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match COMMON_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        // match MILITARY_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match FORMAL_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match ACADEMIC_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match RELIGIOUS_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }

        match record.sex.clone().as_str() {
            "male" => features.push(1f64),
            _ => features.push(0f64),
        };

        // features.push(record.siblings_spouses_aboard.clone() as f64);
        // features.push(record.parents_children_aboard.clone() as f64);
        let family_size = record.parents_children_aboard.clone() as f64 + record.siblings_spouses_aboard.clone() as f64 + 1f64;
        features.push(family_size);
        // match family_size {
        //     1f64 => features.push(0f64),
        //     _ => features.push(1f64),
        // }

        // tmp_training_data.push(ticket_number.parse::<f64>().unwrap());
        features.push(record.fare.unwrap().clone() as f64);

        // if record.cabin_number.is_some() {
        //     let cabin = record.cabin_number.clone().unwrap();
        //     let cabin_regex = Regex::new(r"([A-Z]?)([0-9]*)").unwrap();
        //     let cabin_capture_groups = cabin_regex.captures(cabin.as_str()).unwrap();
        //     let deck = cabin_capture_groups.get(1).map_or("", |m| m.as_str());
        //     let room = cabin_capture_groups.get(2).map_or("", |m| m.as_str());
        //     // info!("cabin: {}: deck: {}, room: {}", cabin, deck, room);
        //
        //     match deck {
        //         "A" => features.push(7f64),
        //         "B" => features.push(6f64),
        //         "C" => features.push(5f64),
        //         "D" => features.push(4f64),
        //         "E" => features.push(3f64),
        //         "F" => features.push(2f64),
        //         "G" => features.push(1f64),
        //         _ => features.push(0f64),
        //     }
        //
        // //has multiple cabins
        // // match cabin.contains(" ") {
        // //     true => tmp_training_data.push(1f64),
        // //     _ => tmp_training_data.push(0f64),
        // // }
        // } else {
        //     // don't know deck
        //     features.push(0f64);
        //     // does not have multiple cabins
        //     // features.push(0f64);
        // }

        match record.embarked.clone().as_str() {
            "S" => features.push(1f64),
            "C" => features.push(2f64),
            "Q" => features.push(3f64),
            _ => features.push(0f64),
        };
        // should be 17
        //info!("tmp_training_data.len(): {}", tmp_training_data.len());
        //info!("tmp_training_data: {:?}", tmp_training_data);

        if feature_size == 0_usize {
            feature_size = features.len();
        }

        training_data.append(&mut features);
    }

    let training_data_matrix = linalg::Matrix::new(training_data.len() / feature_size, feature_size, training_data);
    //info!("training_data_matrix: {:?}", training_data_matrix);
    let mut transformer = Standardizer::default();
    let training_data_transformed = transformer.transform(training_data_matrix).unwrap();
    let training_targets = linalg::Vector::new(targets);

    Ok((training_data_transformed, training_targets, feature_size))
}

pub fn parse_test_data(raw_test_data: &Vec<TitanicTestData>) -> io::Result<(rusty_machine::prelude::Matrix<f64>, usize)> {
    let missing_age_count = raw_test_data.iter().filter(|x| x.age.is_none()).count();
    let age_sum: f64 = raw_test_data.iter().filter(|x| x.age.is_some()).map(|x| x.age.unwrap() as f64).sum();
    let age_mean = age_sum / ((raw_test_data.len() - missing_age_count) as f64);
    debug!(
        "missing_age_count: {} out of {}, age_sum: {}, age_mean: {}",
        missing_age_count,
        raw_test_data.len(),
        age_sum,
        age_mean
    );

    let missing_fare_count = raw_test_data.iter().filter(|x| x.fare.is_none()).count();
    let fare_sum: f64 = raw_test_data.iter().filter(|x| x.fare.is_some()).map(|x| x.fare.unwrap() as f64).sum();
    let fare_mean = fare_sum / ((raw_test_data.len() - missing_fare_count) as f64);
    debug!(
        "missing_fare_count: {} out of {}, fare_sum: {}, fare_mean: {}",
        missing_fare_count,
        raw_test_data.len(),
        fare_sum,
        fare_mean
    );

    let mut feature_size = 0_usize;
    let mut test_data: Vec<_> = Vec::new();
    for record in raw_test_data.iter() {
        let mut features: Vec<f64> = Vec::new();
        //info!("record: {}", record);

        let ticket = record.ticket.clone();
        let ticket_regex = Regex::new(r"(SC/AH Balse |LINE|.* )?([0-9]+)?").unwrap();
        let ticket_capture_groups = ticket_regex.captures(ticket.as_str()).unwrap();
        let ticket_prefix = ticket_capture_groups.get(1).map_or("", |m| m.as_str());
        let ticket_number = ticket_capture_groups.get(2).map_or("", |m| m.as_str());
        //info!("ticket: {}: ticket_prefix: {}, ticket_number: {}", ticket, ticket_prefix, ticket_number);
        if ticket_prefix.eq("LINE") {
            continue;
        }

        //tmp_test_data.push(record.passenger_id.clone() as f64);
        features.push(record.passenger_class.clone() as f64);

        let name = record.name.clone();
        // name = name.replace(&honorific, "");
        // let name_split: Vec<_> = name.split(",").collect();
        // let last_name = name_split.get(0).unwrap();
        // let first_name = name_split.get(1).unwrap();
        //info!("{} {} {}", honorific, first_name.trim(), last_name);
        // leave name out????

        let age = match record.age {
            None => age_mean.clone(),
            _ => record.age.unwrap().clone() as f64,
        };

        features.push(age);
        features.push(record.passenger_class.clone() as f64 * age);

        match (0f64..=16f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (17f64..=32f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (33f64..=48f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (49f64..=99f64).contains(&age) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match (0f64..=16f64).contains(&age) && CHILDREN_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        match COMMON_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
            true => features.push(1f64),
            _ => features.push(0f64),
        }

        // match MILITARY_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match FORMAL_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match ACADEMIC_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }
        //
        // match RELIGIOUS_HONORIFICS.par_iter().any(|x| name.contains(&x.to_string())) {
        //     true => features.push(1f64),
        //     _ => features.push(0f64),
        // }

        match record.sex.clone().as_str() {
            "male" => features.push(1f64),
            _ => features.push(0f64),
        };

        // features.push(record.siblings_spouses_aboard.clone() as f64);
        // features.push(record.parents_children_aboard.clone() as f64);
        let family_size = record.parents_children_aboard.clone() as f64 + record.siblings_spouses_aboard.clone() as f64 + 1f64;
        features.push(family_size);
        // match family_size {
        //     1f64 => features.push(0f64),
        //     _ => features.push(1f64),
        // }

        // tmp_test_data.push(ticket_number.parse::<f64>().unwrap());
        //
        match record.fare {
            None => features.push(fare_mean.clone()),
            _ => features.push(record.fare.unwrap().clone() as f64),
        }

        // if record.cabin_number.is_some() {
        //     let cabin = record.cabin_number.clone().unwrap();
        //     let cabin_regex = Regex::new(r"([A-Z]?)([0-9]*)").unwrap();
        //     let cabin_capture_groups = cabin_regex.captures(cabin.as_str()).unwrap();
        //     let deck = cabin_capture_groups.get(1).map_or("", |m| m.as_str());
        //     let room = cabin_capture_groups.get(2).map_or("", |m| m.as_str());
        //     // info!("cabin: {}: deck: {}, room: {}", cabin, deck, room);
        //
        //     match deck {
        //         "A" => features.push(7f64),
        //         "B" => features.push(6f64),
        //         "C" => features.push(5f64),
        //         "D" => features.push(4f64),
        //         "E" => features.push(3f64),
        //         "F" => features.push(2f64),
        //         "G" => features.push(1f64),
        //         _ => features.push(0f64),
        //     }
        //
        // //has multiple cabins
        // // match cabin.contains(" ") {
        // //     true => tmp_test_data.push(1f64),
        // //     _ => tmp_test_data.push(0f64),
        // // }
        // } else {
        //     // don't know deck
        //     features.push(0f64);
        //     //does not have multiple cabins
        //     //features.push(0f64);
        // }

        match record.embarked.clone().as_str() {
            "S" => features.push(1f64),
            "C" => features.push(2f64),
            "Q" => features.push(3f64),
            _ => features.push(0f64),
        };

        if feature_size == 0_usize {
            feature_size = features.len();
        }

        //info!("tmp_test_data.len(): {}", tmp_test_data.len());
        test_data.append(&mut features);
    }

    let test_data_matrix = linalg::Matrix::new(test_data.len() / feature_size, feature_size, test_data);
    let mut transformer = Standardizer::default();
    let test_data_transformed = transformer.transform(test_data_matrix).unwrap();

    Ok((test_data_transformed, feature_size))
}

pub fn rounded_row_accuracy(outputs: &linalg::Matrix<f64>, targets: &linalg::Matrix<f64>) -> f64 {
    let mut rounded_outputs: Vec<f64> = Vec::new();
    for output in outputs.data().iter() {
        rounded_outputs.push(libm::round(output.clone()));
    }
    let rounded_outputs_matrix = linalg::Matrix::new(outputs.rows(), outputs.cols(), rounded_outputs);
    accuracy(rounded_outputs_matrix.iter_rows(), targets.iter_rows())
}
