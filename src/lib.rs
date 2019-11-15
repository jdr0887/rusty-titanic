#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;

use std::fmt;

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
    pub fare: f32,

    #[serde(rename = "Cabin")]
    pub cabin_number: Option<String>,

    #[serde(rename = "Embarked")]
    pub embarked: String,
}

impl fmt::Display for TitanicTrainingData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TitanicTrainingData(passenger_id: {}, survived: {}, passenger_class: {}, name: {}, sex: {}, age: {:?}, siblings_spouses_aboard: {}, parents_children_aboard: {}, ticket: {}, fare: {}, cabin_number: {:?}, embarked: {})",
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
        fare: f32,
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
    pub static ref COMMON_HONORIFICS: Vec<&'static str> = {
        let mut ret: Vec<_> = Vec::new();
        ret.push("Master.");
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
        ret.push("Jonkheer.");
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
